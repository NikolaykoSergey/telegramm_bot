"""
Telegram бот для работы с технической документацией
ЛОКАЛЬНАЯ ВЕРСИЯ:
- Ollama (qwen2.5:3b) вместо ChatLLM API
- Qdrant вместо FAISS
- PaddleOCR для распознавания текста
- Docling для структурированного извлечения
- Чистка текста через LLM перед индексацией
- Эмбеддинги: paraphrase-multilingual-MiniLM-L12-v2
"""

import logging
import re
import asyncio
from golden_dataset_manager import GoldenDatasetManager
from telegram.ext import ApplicationBuilder
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from local_rag_system import RAGSystem
from local_session_logger import SessionLogger
from local_config import TELEGRAM_BOT_TOKEN, LOG_LEVEL, OLLAMA_MODEL, MAX_HISTORY_CHARS, INITIAL_DATA_FIELDS, check_config

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, LOG_LEVEL),
)
logger = logging.getLogger(__name__)

# Проверка конфигурации
if not check_config():
    exit(1)

# Инициализация RAG системы и логгера сессий
rag_system = RAGSystem()
session_logger = SessionLogger()
golden_dataset = GoldenDatasetManager()


def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы Markdown"""
    if not text:
        return text
    return re.sub(r'([_*`\\\[\]()~>#+\-=|{}.!])', r'\\\1', str(text))


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Создание основной клавиатуры с кнопками"""
    keyboard = [
        [KeyboardButton("📊 Статистика"), KeyboardButton("ℹ️ Справка")],
        [KeyboardButton("🗑️ Сброс истории")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def get_feedback_keyboard() -> ReplyKeyboardMarkup:
    """Клавиатура обратной связи"""
    keyboard = [
        [KeyboardButton("👍 Помог"), KeyboardButton("👎 Не помог")],
    ]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)


def init_user_data(context: ContextTypes.DEFAULT_TYPE):
    """Инициализация данных пользователя"""
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    if 'clarification_questions' not in context.user_data:
        context.user_data['clarification_questions'] = []
    if 'original_query' not in context.user_data:
        context.user_data['original_query'] = None
    if 'initial_data_provided' not in context.user_data:
        context.user_data['initial_data_provided'] = False
    if 'awaiting_initial_data' not in context.user_data:
        context.user_data['awaiting_initial_data'] = {}
    if 'last_bot_response' not in context.user_data:
        context.user_data['last_bot_response'] = None


def trim_history_by_chars(history: list, max_chars: int = MAX_HISTORY_CHARS) -> list:
    """Обрезает историю по символам"""
    if not history:
        return history

    total = 0
    result = []

    for msg in reversed(history):
        text = msg.get("content", "") or ""
        length = len(text)
        if total + length > max_chars and result:
            break
        total += length
        result.append(msg)

    result.reverse()
    return result


def parse_initial_data(text: str) -> dict:
    """Парсит начальные данные из текста"""
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    data = {}

    for line in lines:
        match = re.match(r'^(\d+)[\.\s]+(.+)$', line)
        if match:
            num = int(match.group(1))
            value = match.group(2).strip()
            if 1 <= num <= len(INITIAL_DATA_FIELDS):
                field_name = INITIAL_DATA_FIELDS[num - 1]
                data[field_name] = value

    return data


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    init_user_data(context)
    context.user_data['history'] = []
    context.user_data['clarification_questions'] = []
    context.user_data['original_query'] = None
    context.user_data['initial_data_provided'] = False
    context.user_data['awaiting_initial_data'] = {}
    context.user_data['last_bot_response'] = None

    user = update.effective_user
    session_logger.start_session(user)

    welcome_message = f"""
🤖 Добро пожаловать в бот технической документации!

Я помогу найти информацию в ваших технических документах, используя AI-анализ с RAG.

🔧 Локальная версия:
• LLM: Ollama ({OLLAMA_MODEL})
• Векторная БД: Qdrant
• OCR: PaddleOCR
• Docling: структурированное извлечение
• Эмбеддинги: MiniLM-L12-v2

Для начала работы мне нужны исходные данные по объекту.
"""
    await update.message.reply_text(
        welcome_message,
        reply_markup=get_main_keyboard(),
    )

    # Формируем шаблон с нумерацией
    template_lines = [
        "📋 Пожалуйста, отправьте исходные данные в формате:",
        "",
    ]
    for i, field in enumerate(INITIAL_DATA_FIELDS, 1):
        template_lines.append(f"{i}. {field}")

    template_lines.append("")
    template_lines.append(
        "Пример:\n"
        "1. К-12345\n"
        "2. +79991234567\n"
        "3. OTIS Gen2\n"
        "4. 1.6 м/с\n"
        "5. 16\n"
        "6. 630 кг\n"
        "7. Москва"
    )
    template_lines.append("")
    template_lines.append("После этого можете задавать любые вопросы по документации.")

    template_text = "\n".join(template_lines)
    await update.message.reply_text(template_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = f"""
📚 СПРАВКА ПО ИСПОЛЬЗОВАНИЮ БОТА

🤖 AI-режим:
- Просто напишите свой вопрос
- Бот проанализирует документацию и даст ответ
- При необходимости бот задаст уточняющие вопросы
- Используемая модель: Ollama ({OLLAMA_MODEL})
- Бот помнит контекст беседы (до {MAX_HISTORY_CHARS} символов истории)

🎛 Кнопки:
- 📊 Статистика — список и статистика документов
- ℹ️ Справка — показывает это сообщение
- 🗑️ Сброс истории — очистить память диалога
- 👍 Помог / 👎 Не помог — оценка ответа

⌨️ Команды:
/start — главное меню и новая сессия
/help — справка
/stats — статистика документов
/index — индексация документов
/reindex — полная переиндексация (с нуля)
/continue_index — продолжить индексацию
/stop_index — остановить индексацию
/test — диагностика системы
/reset — сбросить историю диалога

💾 Память диалога:
- Бот помнит до {MAX_HISTORY_CHARS} символов истории
- История сбрасывается при /start, /reset или перезапуске бота
- Все сессии сохраняются в папку sessions/ с полной перепиской

🔧 Технологии:
- LLM: Ollama ({OLLAMA_MODEL}) — локально, бесплатно
- Векторная БД: Qdrant (Docker)
- OCR: PaddleOCR
- Docling: структурированное извлечение
- Эмбеддинги: MiniLM-L12-v2 (384 dim)
- Чистка текста через LLM перед индексацией
"""
    await update.message.reply_text(help_text)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /reset"""
    init_user_data(context)
    user = update.effective_user

    session_logger.add_messages(user, [
        {"role": "system", "content": "[RESET] Пользователь сбросил историю диалога"}
    ])

    context.user_data['history'] = []
    context.user_data['clarification_questions'] = []
    context.user_data['original_query'] = None
    context.user_data['last_bot_response'] = None

    await update.message.reply_text(
        "🗑️ История диалога очищена. Продолжаем работу с чистого листа!",
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    text = update.message.text
    logger.info("🕹 Нажата кнопка: %s", text)

    if text == "📊 Статистика":
        await stats_command(update, context)

    elif text == "ℹ️ Справка":
        await help_command(update, context)

    elif text == "🗑️ Сброс истории":
        await reset_command(update, context)

    elif text == "👍 Помог":
        await handle_feedback_helpful(update, context)

    elif text == "👎 Не помог":
        await handle_feedback_not_helpful(update, context)


async def handle_feedback_helpful(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка положительной обратной связи"""
    user = update.effective_user
    last_response = context.user_data.get("last_bot_response")

    if not last_response:
        await update.message.reply_text(
            "Нет последнего ответа для оценки.",
            reply_markup=get_main_keyboard(),
        )
        return

    session_logger.log_feedback(
        user,
        "helpful",
        f"Q: {last_response['question']}\nA: {last_response['answer']}",
    )

    # ✅ ДОБАВЛЯЕМ В GOLDEN DATASET
    golden_dataset.add_question(
        question=last_response['question'],
        answer=last_response['answer'],
        sources=last_response.get('sources', []),
        user_id=user.id,
        feedback="helpful"
    )

    context.user_data["last_bot_response"] = None

    await update.message.reply_text(
        "Спасибо за обратную связь! 😊\nВопрос добавлен в базу знаний.",
        reply_markup=get_main_keyboard(),
    )


async def handle_feedback_not_helpful(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка отрицательной обратной связи"""
    user = update.effective_user
    last_response = context.user_data.get('last_bot_response')

    if not last_response:
        await update.message.reply_text(
            "Нет последнего ответа для оценки.",
            reply_markup=get_main_keyboard(),
        )
        return

    session_logger.log_feedback(
        user,
        "not_helpful_explicit",
        f"Q: {last_response['question']}\nA: {last_response['answer']}"
    )

    context.user_data['last_bot_response'] = None

    await update.message.reply_text(
        "Понял. Уточни, что именно тебе нужно, постараюсь помочь лучше.",
        reply_markup=get_main_keyboard(),
    )

async def correct_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /correct для корректировки ответа"""
    last_response = context.user_data.get("last_bot_response")

    if not last_response:
        await update.message.reply_text(
            "❌ Нет последнего ответа для корректировки.\n"
            "Сначала задайте вопрос и получите ответ."
        )
        return

    # Сохраняем состояние "ожидание корректировки"
    context.user_data["awaiting_correction"] = True

    await update.message.reply_text(
        f"📝 Корректировка ответа\n\n"
        f"Вопрос: {last_response['question']}\n\n"
        f"Текущий ответ бота:\n{last_response['answer'][:500]}...\n\n"
        f"Напишите ПРАВИЛЬНЫЙ ответ (или /cancel для отмены):"
    )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена текущей операции"""
    context.user_data["awaiting_correction"] = False
    context.user_data["clarification_questions"] = []
    context.user_data["original_query"] = None

    await update.message.reply_text(
        "❌ Операция отменена.",
        reply_markup=get_main_keyboard()
    )

async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /reindex"""
    if rag_system.is_indexing():
        await update.message.reply_text(
            "⚠️ Индексация уже выполняется. Остановите её сначала.",
        )
        return

    await update.message.reply_text("🔄 Начинаю полную переиндексацию документов (с нуля)...")

    asyncio.create_task(run_indexing(update, context, continue_indexing=False))


async def continue_index_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /continue_index"""
    if rag_system.is_indexing():
        await update.message.reply_text(
            "⚠️ Индексация уже выполняется.",
        )
        return

    await update.message.reply_text("🔄 Начинаю индексацию документов (продолжение)...")

    asyncio.create_task(run_indexing(update, context, continue_indexing=True))


async def index_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /index"""
    await continue_index_command(update, context)


async def stop_index_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stop_index"""
    if not rag_system.is_indexing():
        await update.message.reply_text(
            "⚠️ Индексация не выполняется.",
        )
        return

    rag_system.stop_indexing_process()
    await update.message.reply_text(
        "🛑 Запрос на остановку индексации отправлен. Подождите завершения текущего файла...",
    )


async def run_indexing(update: Update, context: ContextTypes.DEFAULT_TYPE, continue_indexing: bool = True):
    """Запуск индексации в фоновом режиме"""
    try:
        await asyncio.to_thread(rag_system.index_documents, continue_indexing=continue_indexing)

        stats = rag_system.get_stats()

        await update.message.reply_text(
            f"✅ Индексация завершена!\n\n"
            f"📊 Статистика:\n"
            f"• Файлов проиндексировано: {stats['indexed_files_count']}\n"
            f"• Фрагментов в базе: {stats['total_documents']}\n"
            f"• Размерность векторов: {stats['vector_size']}",
        )
    except Exception as e:
        logger.error("Ошибка при индексации: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка при индексации: {str(e)}",
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    query = (update.message.text or "").strip()
    if not query:
        return

    init_user_data(context)
    user = update.effective_user

    logger.info("💬 Текстовый запрос пользователя: %s", query)

    # ✅ ПРОВЕРКА: ожидается ли корректировка
    if context.user_data.get("awaiting_correction", False):
        await handle_correction_input(update, context, query)
        return

    # 1) Если стартовые данные ещё не получены
    if not context.user_data.get('initial_data_provided', False):
        data = parse_initial_data(query)

        if not data or len(data) < 3:
            await update.message.reply_text(
                "⚠️ Не удалось распознать данные. Пожалуйста, используйте формат:\n\n"
                "1. Номер контракта\n"
                "2. Номер телефона\n"
                "3. Модель лифта\n"
                "...\n\n"
                "Попробуйте ещё раз."
            )
            return

        context.user_data['initial_data_provided'] = True
        session_logger.set_initial_data(user, data)

        confirmation = "✅ Исходные данные получены:\n\n"
        for field, value in data.items():
            confirmation += f"• {field}: {value}\n"
        confirmation += "\nТеперь можете задавать вопросы по документации."

        await update.message.reply_text(confirmation)
        return

    # 2) Если пользователь пишет после ответа — значит ответ был нерелевантным
    if context.user_data.get('last_bot_response'):
        last_resp = context.user_data['last_bot_response']
        session_logger.log_feedback(
            user,
            "not_helpful_implicit",
            f"Пользователь продолжил диалог после ответа.\nQ: {last_resp['question']}\nA: {last_resp['answer']}"
        )
        context.user_data['last_bot_response'] = None

    # 3) Проверяем уточняющие вопросы
    clarification_questions = context.user_data.get('clarification_questions', [])
    original_query = context.user_data.get('original_query')

    if clarification_questions and original_query:
        await handle_clarification_response(update, context, query)
        return

    # 4) Обычный AI-поиск
    await perform_ai_search(update, context, query)


async def handle_clarification_response(update: Update, context: ContextTypes.DEFAULT_TYPE, response: str):
    """Обработка ответа на уточняющий вопрос"""
    clarification_questions = context.user_data.get('clarification_questions', [])
    original_query = context.user_data.get('original_query')

    if response.isdigit():
        question_num = int(response)
        if 1 <= question_num <= len(clarification_questions):
            selected_question = clarification_questions[question_num - 1]

            context.user_data['clarification_questions'] = []
            context.user_data['original_query'] = None

            refined_query = f"{original_query}. {selected_question}"

            await update.message.reply_text(
                f"✅ Понял! Ищу информацию по теме: {selected_question}",
            )

            await perform_ai_search(update, context, refined_query, skip_clarification=True)
        else:
            await update.message.reply_text(
                f"❌ Неверный номер. Выберите от 1 до {len(clarification_questions)}",
            )
    else:
        context.user_data['clarification_questions'] = []
        context.user_data['original_query'] = None

        await update.message.reply_text(
            "✅ Понял, ищу по вашему уточнению...",
        )

        await perform_ai_search(update, context, response, skip_clarification=True)

async def handle_correction_input(update: Update, context: ContextTypes.DEFAULT_TYPE, corrected_answer: str):
    """Обработка ввода правильного ответа"""
    user = update.effective_user
    last_response = context.user_data.get("last_bot_response")

    if not last_response:
        await update.message.reply_text("❌ Ошибка: нет сохранённого ответа.")
        context.user_data["awaiting_correction"] = False
        return

    # Сохраняем в golden dataset с правильным ответом
    golden_dataset.add_question(
        question=last_response['question'],
        answer=corrected_answer,  # ← правильный ответ от пользователя
        sources=last_response.get('sources', []),
        user_id=user.id,
        feedback="corrected"
    )

    # Логируем в сессию
    session_logger.add_messages(
        user,
        [
            {"role": "system", "content": f"[CORRECTION] Пользователь исправил ответ:\nВопрос: {last_response['question']}\nБыло: {last_response['answer'][:200]}...\nСтало: {corrected_answer}"}
        ]
    )

    context.user_data["awaiting_correction"] = False
    context.user_data["last_bot_response"] = None

    await update.message.reply_text(
        "✅ Спасибо! Правильный ответ сохранён в базу знаний.\n"
        "Это поможет улучшить качество ответов в будущем.",
        reply_markup=get_main_keyboard()
    )


async def perform_ai_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, skip_clarification: bool = False):
    """Выполнение поиска с использованием AI (RAG)"""
    user = update.effective_user

    await update.message.reply_text(
        f"🤖 Анализирую документацию...\nЗапрос: {query}",
    )

    try:
        if not skip_clarification:
            questions = rag_system.generate_clarification_questions(query)

            if questions and len(questions) > 0:
                context.user_data['clarification_questions'] = questions
                context.user_data['original_query'] = query

                response = "❓ Уточните, пожалуйста, что именно вас интересует:\n\n"
                for i, question in enumerate(questions, 1):
                    response += f"{i}. {question}\n"

                response += "\nВведите номер вопроса или напишите свой уточняющий запрос"

                await update.message.reply_text(response)
                return

        # Получаем историю
        history = context.user_data.get('history', [])

        # Выполняем RAG-запрос с историей
        result = rag_system.query_with_history(history, query)

        raw_answer = result['answer']
        sources = result.get('sources', [])
        relevance = result.get('relevance', 0.0)

        # Сохраняем для обратной связи
        context.user_data['last_bot_response'] = {
            'question': query,
            'answer': raw_answer
        }

        response = f"💡 Ответ:\n\n{raw_answer}\n\n"
        response += f"📊 Релевантность: {relevance:.1f}%\n\n"

        if sources:
            response += "📚 Источники:\n"
            for i, source in enumerate(sources, 1):
                file_name = str(source.get('file', ''))
                page = str(source.get('page', ''))
                score = source.get('score', 0)
                response += f"{i}. {file_name} (стр. {page}, релевантность: {score:.2f})\n"

        # Сохраняем в оперативную историю
        context.user_data['history'].append({"role": "user", "content": query})
        context.user_data['history'].append({"role": "assistant", "content": raw_answer})

        # Обрезаем историю по символам
        context.user_data['history'] = trim_history_by_chars(context.user_data['history'], MAX_HISTORY_CHARS)

        # Параллельно пишем в файл сессии
        session_logger.add_messages(user, [
            {"role": "user", "content": query},
            {"role": "assistant", "content": raw_answer},
        ])

        if len(response) > 4000:
            parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]
            for part in parts:
                await update.message.reply_text(part, reply_markup=get_feedback_keyboard())
        else:
            await update.message.reply_text(response, reply_markup=get_feedback_keyboard())

    except Exception as e:
        logger.error("Ошибка при AI поиске: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка при обработке запроса: {str(e)}",
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stats"""
    try:
        stats = rag_system.get_stats()

        if stats['total_documents'] == 0:
            await update.message.reply_text(
                "📊 Индекс пуст.\n\n"
                "Добавьте документы в папку documents/ и выполните /index",
            )
            return

        response = "📊 СТАТИСТИКА ДОКУМЕНТОВ\n\n"
        response += f"📁 Проиндексировано файлов: {stats['indexed_files_count']}\n"
        response += f"📄 Всего фрагментов: {stats['total_documents']}\n"
        response += f"🔢 Размерность векторов: {stats['vector_size']}\n\n"

        response += "📚 Проиндексированные файлы:\n"
        indexed_list = stats.get('indexed_files_list', [])
        for i, file in enumerate(indexed_list[:15], 1):
            response += f"{i}. {file}\n"

        if len(indexed_list) > 15:
            response += f"\n...и ещё {len(indexed_list) - 15} файлов\n"

        await update.message.reply_text(response)

    except Exception as e:
        logger.error("Ошибка при получении статистики: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка: {str(e)}",
        )


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /test"""
    await update.message.reply_text(
        "🔌 Запуск диагностики системы...\nЭто может занять несколько секунд",
    )

    try:
        test_result = rag_system.test_connection()
        await update.message.reply_text(
            test_result['message'],
        )

    except Exception as e:
        logger.error("Ошибка при выполнении диагностики: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка при выполнении диагностики:\n{str(e)}",
        )

async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный логгер всех апдейтов (для отладки)"""
    try:
        user = update.effective_user
        uid = user.id if user else "unknown"
        uname = user.username if user and user.username else "no_username"
        text = None

        if update.message and update.message.text:
            text = update.message.text
        elif update.callback_query and update.callback_query.data:
            text = f"[callback] {update.callback_query.data}"

        logger.info("📥 Update от %s (%s): %s", uid, uname, repr(text))
    except Exception as e:
        logger.error("Ошибка в log_update: %s", repr(e))

def main():
    """Запуск бота"""
    logger.info("🚀 Запуск бота (локальная версия)...")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Глобальный логгер апдейтов
    application.add_handler(MessageHandler(filters.ALL, log_update), group=-1)

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("index", index_command))
    application.add_handler(CommandHandler("reindex", reindex_command))
    application.add_handler(CommandHandler("continue_index", continue_index_command))
    application.add_handler(CommandHandler("stop_index", stop_index_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(CommandHandler("correct", correct_command))
    application.add_handler(CommandHandler("cancel", cancel_command))

    # Кнопки
    application.add_handler(
        MessageHandler(
            filters.Regex(r"^(📊 Статистика|ℹ️ Справка|🗑️ Сброс истории|👍 Помог|👎 Не помог)$"),
            handle_button,
        )
    )

    # Любой другой текст
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("✅ Бот успешно запущен!")
    logger.info("📊 Модель: Ollama (%s)", OLLAMA_MODEL)
    logger.info("💾 Память диалога: до %d символов", MAX_HISTORY_CHARS)
    logger.info("📁 Сессии сохраняются в папку: sessions/")

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

#