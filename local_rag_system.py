import logging
import json
import time
import psutil
import re
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm

from local_config import (
    DOCUMENTS_FOLDER,
    TOP_K_RESULTS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)
from local_document_processor import DocumentProcessor
from local_vector_store import VectorStore
from local_ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG система: индексация + поиск + генерация ответов"""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()

        self.ollama = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE,
        )

        self.indexed_files_path = Path("indexed_files.json")
        self.indexed_files = self._load_indexed_files()

        self._indexing = False
        self._stop_indexing = False

        logger.info("✅ RAGSystem инициализирован")

    # ==============================
    # ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
    # ==============================

    def _load_indexed_files(self) -> List[str]:
        """Загрузка списка проиндексированных файлов"""
        if self.indexed_files_path.exists():
            try:
                with open(self.indexed_files_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("indexed_files", [])
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки indexed_files.json: {repr(e)}")
        return []

    def _save_indexed_files(self):
        """Сохранение списка проиндексированных файлов"""
        try:
            with open(self.indexed_files_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"indexed_files": self.indexed_files},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения indexed_files.json: {repr(e)}")

    @staticmethod
    def _is_general_chat(query: str) -> bool:
        """Проверка на общий/болтовой вопрос, не про технику"""
        text = query.lower()

        general_keywords = [
            "как тебя зовут",
            "как твое имя",
            "как твоё имя",
            "что ты умеешь",
            "кто ты",
            "расскажи анекдот",
            "анекдот",
            "шутку расскажи",
            "шутка",
            "кто тебя создал",
            "что ты такое",
            "что ты можешь",
        ]

        return any(kw in text for kw in general_keywords)

    @staticmethod
    def _is_elevator_related(query: str) -> bool:
        """
        Примитивная эвристика: запрос про лифты или нет.
        По умолчанию считаем, что если есть "плата", "настройка", "ошибка" и т.п. — это лифты.
        """
        text = query.lower()

        elevator_keywords = [
            "лифт",
            "кабина",
            "шахта",
            "ограничитель скорости",
            "станция управления",
            "станция упр.",
            "су-к",
            "сук-",
            "мс-1",
            "ms-1",
            "ms1",
            "лебедка",
            "лебйдка",
            "door",
            "elevator",
        ]

        tech_action_keywords = [
            "плата",
            "board",
            "контроллер",
            "блок",
            "модуль",
            "настройка",
            "настроить",
            "конфигурация",
            "параметр",
            "параметры",
            "ошибка",
            "fault",
            "alarm",
            "как сделать",
            "как настроить",
            "как подключить",
            "подключение",
            "дип-переключатель",
            "dip",
            "jumper",
            "перемычка",
        ]

        for kw in elevator_keywords + tech_action_keywords:
            if kw in text:
                return True

        return False

    @staticmethod
    def _answer_needs_clarification(answer: str) -> bool:
        """
        Проверяем, содержит ли ответ фразы типа "нет информации", "уточните" и т.п.
        Если да — значит, нужно сгенерировать уточняющие вопросы.
        """
        text = answer.lower()

        unclear_phrases = [
            "нет точной информации",
            "не найдено",
            "не хватает данных",
            "уточните",
            "какую модель",
            "какую плату",
            "какой режим",
            "не указано",
            "не ясно",
            "недостаточно информации",
            "не могу ответить",
        ]

        return any(phrase in text for phrase in unclear_phrases)

    # ==============================
    # ИНДЕКСАЦИЯ
    # ==============================

    def index_documents(self, continue_indexing: bool = True):
        """
        Индексация документов из папки documents/
        """
        if self._indexing:
            logger.warning("⚠️ Индексация уже выполняется")
            return

        self._indexing = True
        self._stop_indexing = False

        process_start = time.time()
        process = psutil.Process()

        try:
            if not continue_indexing:
                logger.info("🔄 НАЧИНАЮ ПОЛНУЮ ПЕРЕИНДЕКСАЦИЮ (очистка БД)...")
                self.vector_store.clear_collection()
                self.indexed_files = []
                logger.info("✅ Векторная БД очищена")

            files = list(DOCUMENTS_FOLDER.glob("*.pdf")) + list(
                DOCUMENTS_FOLDER.glob("*.docx")
            )

            if not files:
                logger.warning(f"⚠️ Нет файлов в папке {DOCUMENTS_FOLDER}")
                return

            if continue_indexing:
                files_before = len(files)
                files = [f for f in files if f.name not in self.indexed_files]
                files_after = len(files)
                logger.info(
                    f"📊 Файлов до фильтрации: {files_before}, после: {files_after}"
                )

            if not files:
                logger.info("✅ Все файлы уже проиндексированы")
                return

            logger.info(f"📚 НАЙДЕНО ФАЙЛОВ ДЛЯ ИНДЕКСАЦИИ: {len(files)}")
            logger.info(f"📊 Начинаю индексацию (continue={continue_indexing})")
            logger.info(
                f"💾 Память в начале: {process.memory_info().rss / 1024 / 1024:.1f}MB"
            )

            total_fragments = 0
            total_files_processed = 0
            failed_files = []

            for file_idx, file_path in enumerate(
                tqdm(files, desc="Индексация", unit="файл"), 1
            ):
                if self._stop_indexing:
                    logger.info("🛑 Индексация остановлена пользователем")
                    break

                logger.info("\n" + "=" * 80)
                logger.info(f"📁 ФАЙЛ {file_idx}/{len(files)}: {file_path.name}")
                logger.info("=" * 80)
                file_start = time.time()

                try:
                    logger.info("📄 Начинаю обработку файла...")
                    fragments = self.document_processor.process_file(file_path)

                    if not fragments:
                        logger.warning(
                            f"⚠️ Файл {file_path.name} не содержит текста"
                        )
                        failed_files.append((file_path.name, "нет текста"))
                        continue

                    logger.info(
                        f"📤 Загрузка {len(fragments)} фрагментов в векторную БД..."
                    )
                    db_start = time.time()
                    self.vector_store.add_documents(fragments)
                    db_time = time.time() - db_start

                    self.indexed_files.append(file_path.name)
                    self._save_indexed_files()

                    total_fragments += len(fragments)
                    total_files_processed += 1

                    file_time = time.time() - file_start
                    memory_usage = process.memory_info().rss / 1024 / 1024

                    logger.info(f"✅ ФАЙЛ {file_path.name} УСПЕШНО ПРОИНДЕКСИРОВАН:")
                    logger.info(f"   📊 Фрагментов: {len(fragments)}")
                    logger.info(
                        f"   ⏱️ Время обработки файла: {file_time:.1f}с"
                    )
                    logger.info(
                        f"   ⏱️ Время загрузки в БД: {db_time:.1f}с"
                    )
                    logger.info(f"   💾 Память: {memory_usage:.1f}MB")
                    if file_time > 0:
                        logger.info(
                            f"   📈 Скорость: {len(fragments) / file_time:.1f} фрагм/сек"
                        )

                except Exception as e:
                    logger.error(
                        f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ОБРАБОТКЕ ФАЙЛА {file_path.name}: {repr(e)}"
                    )
                    failed_files.append((file_path.name, str(e)))
                    continue

            total_time = time.time() - process_start
            final_memory = process.memory_info().rss / 1024 / 1024

            logger.info("\n" + "=" * 80)
            logger.info("🎉 ИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
            logger.info("=" * 80)
            logger.info("📊 ИТОГОВАЯ СТАТИСТИКА:")
            logger.info(f"   📁 Всего файлов для индексации: {len(files)}")
            logger.info(f"   ✅ Успешно обработано: {total_files_processed}")
            logger.info(f"   ❌ Не удалось обработать: {len(failed_files)}")
            logger.info(f"   📄 Всего фрагментов: {total_fragments}")
            logger.info(f"   ⏱️ Общее время: {total_time:.1f}с")
            if total_time > 0:
                logger.info(
                    f"   📈 Средняя скорость: {total_fragments / total_time:.1f} фрагм/сек"
                )
            logger.info(f"   💾 Память после: {final_memory:.1f}MB")

            if total_files_processed > 0:
                logger.info(
                    f"   📊 Среднее на файл: {total_fragments / total_files_processed:.1f} фрагментов"
                )

            if failed_files:
                logger.warning("\n⚠️ НЕ УДАЛОСЬ ОБРАБОТАТЬ ФАЙЛЫ:")
                for file_name, error in failed_files:
                    logger.warning(f"   • {file_name}: {error}")

        except Exception as e:
            logger.error(
                f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ИНДЕКСАЦИИ: {repr(e)}"
            )
            raise

        finally:
            self._indexing = False
            self._stop_indexing = False

    def is_indexing(self) -> bool:
        return self._indexing

    def stop_indexing_process(self):
        self._stop_indexing = True

    # ==============================
    # ОСНОВНОЙ ЗАПРОС (БЕЗ ИСТОРИИ)
    # ==============================

    def query(self, user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Поиск + генерация ответа + автогенерация уточняющих вопросов при необходимости
        """
        logger.info(f"💬 ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}")

        # 0. Общий болтовой вопрос — отвечаем без RAG
        if self._is_general_chat(user_query):
            logger.info("💬 Обнаружен общий вопрос (не про лифты) — отвечаем без RAG")
            answer = self.ollama.generate(
                prompt=f"Пользователь спросил: {user_query}\nОтветь дружелюбно на русском языке.",
                system_prompt="Ты дружелюбный ассистент. Можно чуть пошутить, но без жёсткого мата.",
                max_tokens=256,
            )
            return {"answer": answer, "sources": [], "relevance": 0.0, "clarification_questions": []}

        # 1. Проверяем, похоже ли на лифтовую/технич. тему
        is_elevator = self._is_elevator_related(user_query)
        if not is_elevator:
            logger.info(
                "ℹ️ Запрос не похож на лифтовую тематику. Отвечаем как общий ассистент без RAG."
            )
            answer = self.ollama.generate(
                prompt=f"Пользователь спросил: {user_query}\nОтветь кратко и по делу на русском языке.",
                system_prompt="Ты общий ассистент. Можно использовать любые свои знания.",
                max_tokens=256,
            )
            return {"answer": answer, "sources": [], "relevance": 0.0, "clarification_questions": []}

        # 2. Вопрос технический по лифтам — включаем RAG
        logger.info("🔍 Поиск релевантных документов...")
        documents = self.vector_store.search(user_query, top_k=top_k)

        if not documents:
            logger.warning(
                f"⚠️ Не удалось найти подходящие документы для запроса: {user_query}"
            )
            return {
                "answer": "❌ В проиндексированной документации не найдено ничего по этому вопросу.",
                "sources": [],
                "relevance": 0.0,
                "clarification_questions": [],
            }

        avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
        relevance_percent = avg_score * 100

        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[Источник {idx}: {doc['file']}, стр. {doc['page']}]\n{doc['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # УСИЛЕННЫЙ ПРОМПТ с chain-of-thought
        system_prompt = """Ты — AI-ассистент для работы с технической документацией по лифтам и лифтовому оборудованию.

ТВОЯ ГЛАВНАЯ ЗАДАЧА:
- Максимально полезно ответить на технический вопрос пользователя, используя предоставленный контекст из документации.

ПРАВИЛА РАБОТЫ С КОНТЕКСТОМ:
1. ВНИМАТЕЛЬНО ПРОЧИТАЙ весь предоставленный контекст.
2. Если в контексте есть ЛЮБАЯ релевантная информация (таблицы, параметры, инструкции, описания плат, ошибок, режимов) — ОБЯЗАТЕЛЬНО используй её в ответе.
3. Если есть таблица или список параметров — ВЫПИШИ их явно, структурированно.
4. Если есть пошаговая инструкция — передай её по шагам.
5. НЕ ГОВОРИ "нет информации", если в контексте хоть что-то есть по теме. Лучше дай частичный ответ на основе того, что есть.

ЕСЛИ РЕАЛЬНО НЕТ ДАННЫХ:
- Только если в контексте ВООБЩЕ НЕТ ничего по вопросу — тогда честно скажи:
  "В предоставленных фрагментах документации нет информации по этому вопросу."
- И предложи пользователю уточнить: модель платы, код ошибки, режим работы и т.п.

ФОРМАТ ОТВЕТА:
- Кратко, структурированно, по делу.
- Если есть числа/параметры/адреса — указывай их явно.
- Отвечай на русском языке.

КРИТИЧЕСКИ ВАЖНО:
- НЕ ПРИДУМЫВАЙ данных, которых нет в контексте.
- НЕ подменяй одно устройство другим.
- Но ИСПОЛЬЗУЙ всё, что есть в контексте, максимально полезно."""

        prompt = f"""Контекст из документации:

{context}

Вопрос пользователя:
{user_query}

Твой ответ (сначала проанализируй контекст, потом дай структурированный ответ):"""

        logger.info("🤖 Генерация ответа через LLM на основе документации...")
        answer = self.ollama.generate(
            prompt, system_prompt=system_prompt, max_tokens=1500
        )

        sources = [
            {
                "file": doc["file"],
                "page": doc["page"],
                "score": round(doc["score"], 3),
            }
            for doc in documents
        ]

        logger.info(f"✅ Ответ сгенерирован, релевантность: {relevance_percent:.1f}%")

        # 3. Проверяем, нужны ли уточняющие вопросы
        clarification_questions = []
        if self._answer_needs_clarification(answer):
            logger.info("❓ Ответ требует уточнения — генерируем уточняющие вопросы...")
            clarification_questions = self.generate_clarification_questions(user_query)

        return {
            "answer": answer,
            "sources": sources,
            "relevance": relevance_percent,
            "clarification_questions": clarification_questions,
        }

    # ==============================
    # ЗАПРОС С ИСТОРИЕЙ
    # ==============================

    def query_with_history(
        self, history: List[Dict], user_query: str, top_k: int = TOP_K_RESULTS
    ) -> Dict:
        logger.info(f"💬 ЗАПРОС С ИСТОРИЕЙ: {user_query}")
        logger.info(f"📊 Размер истории: {len(history)} сообщений")

        # Можно использовать те же эвристики (общий/лифтовый)
        if self._is_general_chat(user_query) and not self._is_elevator_related(
            user_query
        ):
            logger.info(
                "💬 Общий вопрос с историей (но не лифтовый) — отвечаем без RAG"
            )
            history_text = ""
            if history:
                lines = []
                for msg in history[-10:]:
                    role = msg.get("role", "user")
                    content = (msg.get("content") or "").strip()
                    if not content:
                        continue
                    prefix = "Пользователь:" if role == "user" else "Ассистент:"
                    lines.append(f"{prefix} {content}")
                if lines:
                    history_text = "\n".join(lines)

            prompt = "История диалога:\n" + history_text + "\n\n" if history_text else ""
            prompt += f"Текущий вопрос пользователя:\n{user_query}\n\nОтветь дружелюбно на русском языке."

            answer = self.ollama.generate(
                prompt,
                system_prompt="Ты разговорный ассистент, можно немного шутить.",
                max_tokens=256,
            )
            return {"answer": answer, "sources": [], "relevance": 0.0, "clarification_questions": []}

        # Лифтовая тема — RAG
        logger.info("🔍 Поиск релевантных документов (с историей)...")
        documents = self.vector_store.search(user_query, top_k=top_k)

        if not documents:
            logger.warning(
                f"⚠️ Не удалось найти подходящие документы для запроса: {user_query}"
            )
            return {
                "answer": "❌ В проиндексированной документации не найдено ничего по этому вопросу.",
                "sources": [],
                "relevance": 0.0,
                "clarification_questions": [],
            }

        avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
        relevance_percent = avg_score * 100

        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[Источник {idx}: {doc['file']}, стр. {doc['page']}]\n{doc['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        history_text = ""
        if history:
            lines = []
            for msg in history[-10:]:
                role = msg.get("role", "user")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                prefix = "Пользователь:" if role == "user" else "Ассистент:"
                lines.append(f"{prefix} {content}")
            if lines:
                history_text = "\n".join(lines)
                logger.info(
                    f"📜 Используется история из {len(lines)} сообщений"
                )

        system_prompt = """Ты — AI-ассистент для работы с технической документацией по лифтам и лифтовому оборудованию.

ТВОЯ ГЛАВНАЯ ЗАДАЧА:
- Максимально полезно ответить на технический вопрос, используя контекст из документации и историю диалога.

ПРАВИЛА:
1. ВНИМАТЕЛЬНО изучи контекст и историю.
2. Если в контексте есть релевантная информация — используй её полностью.
3. Если есть таблицы/параметры/инструкции — выпиши их явно.
4. НЕ ГОВОРИ "нет информации", если хоть что-то есть в контексте.
5. Если данных реально нет — предложи уточнить модель/плату/режим/код ошибки.

ФОРМАТ:
- Кратко, структурированно, по делу.
- На русском языке.

КРИТИЧЕСКИ ВАЖНО:
- НЕ ПРИДУМЫВАЙ данных.
- Но ИСПОЛЬЗУЙ всё, что есть в контексте."""

        prompt_parts = []
        if history_text:
            prompt_parts.append(f"История диалога:\n{history_text}\n")
        prompt_parts.append(f"Контекст из документации:\n{context}\n")
        prompt_parts.append(f"Текущий вопрос пользователя:\n{user_query}\n")
        prompt_parts.append("Твой ответ:")

        prompt = "\n".join(prompt_parts)

        logger.info("🤖 Генерация ответа через LLM с учётом истории...")
        answer = self.ollama.generate(
            prompt, system_prompt=system_prompt, max_tokens=1500
        )

        sources = [
            {
                "file": doc["file"],
                "page": doc["page"],
                "score": round(doc["score"], 3),
            }
            for doc in documents
        ]

        logger.info(f"✅ Ответ сгенерирован, релевантность: {relevance_percent:.1f}%")

        # Проверяем, нужны ли уточняющие вопросы
        clarification_questions = []
        if self._answer_needs_clarification(answer):
            logger.info("❓ Ответ требует уточнения — генерируем уточняющие вопросы...")
            clarification_questions = self.generate_clarification_questions(user_query)

        return {
            "answer": answer,
            "sources": sources,
            "relevance": relevance_percent,
            "clarification_questions": clarification_questions,
        }

    # ==============================
    # УТОЧНЯЮЩИЕ ВОПРОСЫ + СТАТИСТИКА
    # ==============================

    def generate_clarification_questions(self, user_query: str, max_questions: int = 3) -> List[str]:
        """Генерация уточняющих вопросов"""
        system_prompt = """Ты помощник технической поддержки по лифтам.
Пользователь задал вопрос, но для точного ответа не хватает деталей.
Сгенерируй 2-3 коротких уточняющих вопроса (каждый на отдельной строке).
Вопросы должны быть ОЧЕНЬ короткими (максимум 5-7 слов).
Спрашивай про: модель платы, код ошибки, режим работы, тип лифта и т.п."""

        prompt = f"""Вопрос пользователя: {user_query}

Уточняющие вопросы:"""

        try:
            logger.info(
                f"❓ Генерация уточняющих вопросов для: {user_query}"
            )
            response = self.ollama.generate(
                prompt, system_prompt=system_prompt, max_tokens=256
            )

            questions = [
                q.strip() for q in response.split("\n") if q.strip()
            ]
            questions = [
                q.lstrip("0123456789.-) ") for q in questions
            ]  # убираем нумерацию
            questions = [
                q for q in questions if q and len(q.split()) <= 10
            ]

            # Ограничиваем количество
            questions = questions[:max_questions]

            logger.info(
                f"✅ Сгенерировано {len(questions)} уточняющих вопросов"
            )
            return questions

        except Exception as e:
            logger.error(
                f"❌ Ошибка генерации уточняющих вопросов: {repr(e)}"
            )
            return []

    def get_stats(self) -> Dict:
        """Статистика системы"""
        vector_stats = self.vector_store.get_stats()

        return {
            "indexed_files_count": len(self.indexed_files),
            "indexed_files_list": self.indexed_files,
            "total_documents": vector_stats.get("total_documents", 0),
            "vector_size": vector_stats.get("vector_size", 0),
            "embedding_model": vector_stats.get("model", "unknown"),
        }

    def test_connection(self) -> Dict[str, str]:
        """Тест всех компонентов"""
        results = []

        try:
            ok = self.ollama.test_connection()
            if ok:
                results.append(
                    f"✅ Ollama: подключение успешно, модель {self.ollama.model} доступна."
                )
            else:
                results.append(
                    f"❌ Ollama: не удалось подтвердить доступность модели {self.ollama.model}."
                )
        except Exception as e:
            results.append(f"❌ Ollama: ошибка подключения: {repr(e)}")

        try:
            qdrant_test = self.vector_store.test_connection()
            if isinstance(qdrant_test, dict) and "message" in qdrant_test:
                results.append(qdrant_test["message"])
            else:
                results.append(f"ℹ️ Qdrant: {qdrant_test}")
        except Exception as e:
            results.append(f"❌ Qdrant: ошибка подключения: {repr(e)}")

        return {"message": "\n\n".join(results)}