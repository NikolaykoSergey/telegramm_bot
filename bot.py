"""
Telegram бот для работы с технической документацией
Версия 2.0:
- Память диалога с ограничением по символам
- Логирование сессий с начальными данными
- Упрощённый /start без лишней справки
"""

import logging
import re
import asyncio
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from rag_system import RAGSystem
from session_logger import SessionLogger
from config import TELEGRAM_BOT_TOKEN, LOG_LEVEL, CHATLLM_MODEL, check_config

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

# Максимальное количество символов в истории (для контроля расхода кредитов)
MAX_HISTORY_CHARS = 6000

# Поля для начальных данных
INITIAL_DATA_FIELDS = [
    "Номер контракта",
    "Номер телефона",
    "Модель лифта",
    "Скорость",
    "Количество этажей",
    "Грузоподъёмность",
    "Город",
]


def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы Markdown (если когда‑нибудь решим его включить)."""
    if not text:
        return text
    return re.sub(r'([_*`\\\[\]()~>#+\-=|{}.!])', r'\\\1', str(text))


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Создание основной клавиатуры с кнопками."""
    keyboard = [
        [KeyboardButton("🔄 Индексация"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🔌 Тест API"), KeyboardButton("ℹ️ Справка")],
        [KeyboardButton("🗑️ Сброс истории")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def init_user_data(context: ContextTypes.DEFAULT_TYPE):
    """Инициализация данных пользователя."""
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


def trim_history_by_chars(history: list, max_chars: int = MAX_HISTORY_CHARS) -> list:
    """
    Обрезает историю так, чтобы суммарная длина content по символам
    не превышала max_chars. Берём с конца (последние сообщения).
    """
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
    """
    Парсит начальные данные из текста вида:
    1. данные
    2. данные
    ...
    Возвращает словарь {field_name: value}
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    data = {}

    for line in lines:
        # Ищем паттерн: "номер. данные" или "номер данные"
        match = re.match(r'^(\d+)[\.\s]+(.+)$', line)
        if match:
            num = int(match.group(1))
            value = match.group(2).strip()
            if 1 <= num <= len(INITIAL_DATA_FIELDS):
                field_name = INITIAL_DATA_FIELDS[num - 1]
                data[field_name] = value

    return data


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    init_user_data(context)
    context.user_data['history'] = []
    context.user_data['clarification_questions'] = []
    context.user_data['original_query'] = None
    context.user_data['initial_data_provided'] = False
    context.user_data['awaiting_initial_data'] = {}

    user = update.effective_user
    session_logger.start_session(user)

    model_name = CHATLLM_MODEL or "автовыбор через RouteLLM"

    welcome_message = f"""
🤖 Добро пожаловать в бот технической документации!

Я помогу найти информацию в ваших технических документах, используя AI-анализ с RAG.
Текущая модель: {model_name}

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
    """Обработчик команды /help."""
    model_name = CHATLLM_MODEL or "автовыбор через RouteLLM"

    help_text = f"""
📚 СПРАВКА ПО ИСПОЛЬЗОВАНИЮ БОТА

🤖 AI-режим:
- Просто напишите свой вопрос
- Бот проанализирует документацию и даст ответ
- При необходимости бот задаст уточняющие вопросы
- Используемая модель: {model_name}
- Бот помнит контекст беседы (до {MAX_HISTORY_CHARS} символов истории)

🎛 Кнопки:
- 🔄 Индексация — запуск/остановка индексации документов
- 📊 Статистика — список и статистика документов
- 🔌 Тест API — проверка соединения с ChatLLM
- ℹ️ Справка — показывает это сообщение
- 🗑️ Сброс истории — очистить память диалога

⌨️ Команды:
/start — главное меню и новая сессия
/help — справка
/stats — статистика документов
/reindex — полная переиндексация (с нуля)
/continue_index — продолжить индексацию
/stop_index — остановить индексацию
/test — диагностика API
/reset — сбросить историю диалога

💾 Память диалога:
- Бот помнит до {MAX_HISTORY_CHARS} символов истории
- История сбрасывается при /start, /reset или перезапуске бота
- Все сессии сохраняются в папку sessions/ с полной перепиской
"""
    await update.message.reply_text(help_text)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /reset — сброс истории диалога."""
    init_user_data(context)
    user = update.effective_user

    # Записываем событие сброса в лог
    session_logger.add_messages(user, [
        {"role": "system", "content": "[RESET] Пользователь сбросил историю диалога"}
    ])

    context.user_data['history'] = []
    context.user_data['clarification_questions'] = []
    context.user_data['original_query'] = None

    await update.message.reply_text(
        "🗑️ История диалога очищена. Продолжаем работу с чистого листа!",
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик нажатий на кнопки.
    ВАЖНО: здесь только локальные действия, без вызова LLM.
    """
    text = update.message.text
    logger.info("🕹 Нажата кнопка: %s", text)

    if text == "🔄 Индексация":
        if rag_system.is_indexing():
            await stop_index_command(update, context)
        else:
            await continue_index_command(update, context)

    elif text == "📊 Статистика":
        await stats_command(update, context)

    elif text == "🔌 Тест API":
        await test_command(update, context)

    elif text == "ℹ️ Справка":
        await help_command(update, context)

    elif text == "🗑️ Сброс истории":
        await reset_command(update, context)


async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /reindex — полная переиндексация."""
    if rag_system.is_indexing():
        await update.message.reply_text(
            "⚠️ Индексация уже выполняется. Остановите её сначала.",
        )
        return

    await update.message.reply_text("🔄 Начинаю полную переиндексацию документов (с нуля)...")

    asyncio.create_task(run_indexing(update, context, continue_indexing=False))


async def continue_index_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /continue_index — продолжение индексации."""
    if rag_system.is_indexing():
        await update.message.reply_text(
            "⚠️ Индексация уже выполняется.",
        )
        return

    await update.message.reply_text("🔄 Начинаю индексацию документов (продолжение)...")

    asyncio.create_task(run_indexing(update, context, continue_indexing=True))


async def stop_index_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stop_index — остановка индексации."""
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
    """Запуск индексации в фоновом режиме."""
    try:
        await asyncio.to_thread(rag_system.index_documents, continue_indexing=continue_indexing)

        stats = rag_system.get_stats()

        await update.message.reply_text(
            f"✅ Индексация завершена!\n\n"
            f"📊 Статистика:\n"
            f"• Файлов проиндексировано: {stats['indexed_files_count']}\n"
            f"• Фрагментов в базе: {stats['total_documents']}\n"
            f"• Типов контента: {len(stats['types'])}",
        )
    except Exception as e:
        logger.error("Ошибка при индексации: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка при индексации: {str(e)}",
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик текстовых сообщений из поля ввода.
    Всё, что не кнопка и не команда — идёт сюда.
    """
    query = (update.message.text or "").strip()
    if not query:
        return

    init_user_data(context)
    user = update.effective_user

    logger.info("💬 Текстовый запрос пользователя: %s", query)

    # 1) Если стартовые данные ещё не получены — парсим их
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

        # Сохраняем в SessionLogger
        session_logger.set_initial_data(user, data)

        # Формируем красивое подтверждение
        confirmation = "✅ Исходные данные получены:\n\n"
        for field, value in data.items():
            confirmation += f"• {field}: {value}\n"
        confirmation += "\nТеперь можете задавать вопросы по документации."

        await update.message.reply_text(confirmation)
        return

    # 2) Проверяем уточняющие вопросы
    clarification_questions = context.user_data.get('clarification_questions', [])
    original_query = context.user_data.get('original_query')

    if clarification_questions and original_query:
        await handle_clarification_response(update, context, query)
        return

    # 3) Обычный AI-поиск
    await perform_ai_search(update, context, query)


async def handle_clarification_response(update: Update, context: ContextTypes.DEFAULT_TYPE, response: str):
    """Обработка ответа пользователя на уточняющий вопрос."""
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


async def perform_ai_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, skip_clarification: bool = False):
    """Выполнение поиска с использованием AI (RAG) с учётом истории диалога."""
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
        result = rag_system.query_with_history(history, query, top_k=5)

        raw_answer = result['answer']
        sources = result.get('sources', [])

        response = f"💡 Ответ:\n\n{raw_answer}\n\n"

        if sources:
            response += "📚 Источники:\n"
            for i, source in enumerate(sources, 1):
                file_name = str(source.get('file', ''))
                page = str(source.get('page', ''))
                response += f"{i}. {file_name} (стр. {page})\n"

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
                await update.message.reply_text(part)
        else:
            await update.message.reply_text(response)

    except Exception as e:
        logger.error("Ошибка при AI поиске: %s", repr(e))
        await update.message.reply_text(
            f"❌ Ошибка при обработке запроса: {str(e)}",
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stats."""
    try:
        stats = rag_system.get_stats()

        if stats['total_documents'] == 0:
            await update.message.reply_text(
                "📊 Индекс пуст.\n\n"
                "Добавьте документы в папку documents/ и нажмите 🔄 Индексация",
            )
            return

        response = "📊 СТАТИСТИКА ДОКУМЕНТОВ\n\n"
        response += f"📁 Проиндексировано файлов: {stats['indexed_files_count']}\n"
        response += f"📄 Всего фрагментов: {stats['total_documents']}\n\n"

        response += "📋 По типам контента:\n"
        for type_name, count in stats['types'].items():
            emoji = {'text': '📄', 'table': '📊', 'image_ocr': '🖼️'}.get(type_name, '📌')
            response += f"  {emoji} {type_name}: {count}\n"

        response += "\n📚 Проиндексированные файлы:\n"
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
    """Обработчик команды /test — полная диагностика ChatLLM API."""
    await update.message.reply_text(
        "🔌 Запуск диагностики сети и ChatLLM API...\nЭто может занять несколько секунд",
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


def main():
    """Запуск бота."""
    logger.info("🚀 Запуск бота...")

    if not rag_system.load_index():
        logger.info(
            "ℹ️ Индекс не найден или пуст.\n"
            "Добавьте документы в папку 'documents/' и нажмите кнопку '🔄 Индексация'."
        )

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("reindex", reindex_command))
    application.add_handler(CommandHandler("continue_index", continue_index_command))
    application.add_handler(CommandHandler("stop_index", stop_index_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("test", test_command))

    # Кнопки — чисто локальная логика
    application.add_handler(MessageHandler(
        filters.Regex('^(🔄 Индексация|📊 Статистика|🔌 Тест API|ℹ️ Справка|🗑️ Сброс истории)$'),
        handle_button,
    ))

    # Любой другой текст — в RAG/LLM
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("✅ Бот успешно запущен!")
    logger.info("📊 Модель: %s", CHATLLM_MODEL or "автовыбор через RouteLLM")
    logger.info("💾 Память диалога: до %d символов", MAX_HISTORY_CHARS)
    logger.info("📁 Сессии сохраняются в папку: sessions/")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()