import logging
import json
import time
import psutil
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm

from local_config import (
    DOCUMENTS_FOLDER,
    TOP_K_RESULTS,
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
        self.ollama = OllamaClient()

        self.indexed_files_path = Path("indexed_files.json")
        self.indexed_files = self._load_indexed_files()

        self._indexing = False
        self._stop_indexing = False

        logger.info("✅ RAGSystem инициализирован")

    def _load_indexed_files(self) -> List[str]:
        """Загрузка списка проиндексированных файлов"""
        if self.indexed_files_path.exists():
            try:
                with open(self.indexed_files_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("indexed_files", [])
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки indexed_files.json: {repr(e)}")
        return []

    def _save_indexed_files(self):
        """Сохранение списка проиндексированных файлов"""
        try:
            with open(self.indexed_files_path, 'w', encoding='utf-8') as f:
                json.dump({"indexed_files": self.indexed_files}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения indexed_files.json: {repr(e)}")

    def index_documents(self, continue_indexing: bool = True):
        """
        Индексация документов из папки documents/
        """
        import psutil
        import time

        if self._indexing:
            logger.warning("⚠️ Индексация уже выполняется")
            return

        self._indexing = True
        self._stop_indexing = False

        process_start = time.time()
        process = psutil.Process()

        try:
            if not continue_indexing:
                logger.info("🔄 Полная переиндексация (очистка БД)...")
                self.vector_store.clear_collection()
                self.indexed_files = []

            # Получаем список файлов
            files = list(DOCUMENTS_FOLDER.glob("*.pdf")) + list(DOCUMENTS_FOLDER.glob("*.docx"))

            if not files:
                logger.warning(f"⚠️ Нет файлов в папке {DOCUMENTS_FOLDER}")
                return

            # Фильтруем уже проиндексированные
            if continue_indexing:
                files = [f for f in files if f.name not in self.indexed_files]

            if not files:
                logger.info("✅ Все файлы уже проиндексированы")
                return

            logger.info(f"📚 Найдено файлов для индексации: {len(files)}")
            logger.info(f"📊 Начинаю индексацию (continue={continue_indexing})")
            logger.info(f"💾 Память в начале: {process.memory_info().rss / 1024 / 1024:.1f}MB")

            total_fragments = 0
            total_files_processed = 0

            # Обрабатываем файлы
            for file_idx, file_path in enumerate(tqdm(files, desc="Индексация"), 1):
                if self._stop_indexing:
                    logger.info("🛑 Индексация остановлена пользователем")
                    break

                logger.info(f"\n📁 Файл {file_idx}/{len(files)}: {file_path.name}")
                file_start = time.time()

                # Извлекаем фрагменты
                fragments = self.document_processor.process_file(file_path)

                if not fragments:
                    logger.warning(f"⚠️ Файл {file_path.name} не содержит текста")
                    continue

                # Добавляем в векторную БД
                logger.info(f"   📤 Загрузка {len(fragments)} фрагментов в векторную БД...")
                db_start = time.time()
                self.vector_store.add_documents(fragments)
                db_time = time.time() - db_start

                # Помечаем как проиндексированный
                self.indexed_files.append(file_path.name)
                self._save_indexed_files()

                total_fragments += len(fragments)
                total_files_processed += 1

                file_time = time.time() - file_start
                memory_usage = process.memory_info().rss / 1024 / 1024

                logger.info(f"✅ {file_path.name}:")
                logger.info(f"   📊 {len(fragments)} фрагментов")
                logger.info(f"   ⏱️ Время: {file_time:.1f}с (БД: {db_time:.1f}с)")
                logger.info(f"   💾 Память: {memory_usage:.1f}MB")
                logger.info(f"   📈 Средняя скорость: {len(fragments) / file_time:.1f} фрагм/сек")

            total_time = time.time() - process_start
            final_memory = process.memory_info().rss / 1024 / 1024

            logger.info(f"\n🎉 Индексация завершена!")
            logger.info(f"📊 Итоговая статистика:")
            logger.info(f"   📁 Файлов обработано: {total_files_processed}/{len(files)}")
            logger.info(f"   📄 Всего фрагментов: {total_fragments}")
            logger.info(f"   ⏱️ Общее время: {total_time:.1f}с")
            logger.info(f"   📈 Средняя скорость: {total_fragments / total_time:.1f} фрагм/сек")
            logger.info(f"   💾 Память после: {final_memory:.1f}MB")

            if total_files_processed > 0:
                logger.info(f"   📊 Среднее на файл: {total_fragments / total_files_processed:.1f} фрагментов")

        except Exception as e:
            logger.error(f"❌ Критическая ошибка при индексации: {repr(e)}")
            raise

        finally:
            self._indexing = False
            self._stop_indexing = False

    def is_indexing(self) -> bool:
        """Проверка: идёт ли индексация"""
        return self._indexing

    def stop_indexing_process(self):
        """Остановка индексации"""
        self._stop_indexing = True

    def query(self, user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Поиск + генерация ответа

        Args:
            user_query: Запрос пользователя
            top_k: Количество релевантных фрагментов

        Returns:
            {"answer": "ответ", "sources": [...], "relevance": float}
        """
        logger.info(f"💬 Запрос: {user_query}")

        # 1. Поиск релевантных фрагментов
        documents = self.vector_store.search(user_query, top_k=top_k)

        if not documents:
            return {
                "answer": "❌ Не удалось найти подходящие документы.",
                "sources": [],
                "relevance": 0.0
            }

        # Вычисляем среднюю релевантность
        avg_score = sum(doc.get('score', 0) for doc in documents) / len(documents)
        relevance_percent = avg_score * 100

        # 2. Формируем контекст
        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[Источник {idx}: {doc['file']}, стр. {doc['page']}]\n{doc['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # 3. Генерируем ответ через LLM
        system_prompt = """Ты — AI-ассистент для работы с технической документацией по лифтам и лифтовому оборудованию.

        ТВОИ ЖЁСТКИЕ ПРАВИЛА:
        - Отвечай ТОЛЬКО на основе переданного контекста документов.
        - НИКОГДА не используй общие знания, интернет или свои догадки.
        - Если точной информации в контексте НЕТ — прямо так и скажи: 
          "В предоставленных фрагментах документации нет точной информации по этому вопросу."
        - НЕ подменяй одно устройство/плату другим (например, СУК-1 != МС-1). 
          Если в контексте нет слова "СУК-1", не говори, что "речь идёт о СУК-1" или "СУК-1 не используется". 
          Просто скажи, что по "СУК-1" данных нет.
        - Отвечай кратко и по делу, на русском языке.
        - Если есть таблица или явная инструкция по настройке (переключатели, адреса, DIP, параметры) — приведи их словами или структурированно.

        КРИТИЧЕСКИ ВАЖНО:
        - НЕ ПРИДУМЫВАЙ и НЕ ДОГАДЫВАЙСЯ.
        - Лучше честно ответь "Документация по <термин> в найденных фрагментах не представлена", чем написать чушь."""

        prompt = f"""Контекст из документации:

{context}

Вопрос пользователя:
{user_query}

Ответ:"""

        answer = self.ollama.generate(prompt, system_prompt=system_prompt)

        # 4. Формируем источники
        sources = [
            {
                "file": doc["file"],
                "page": doc["page"],
                "score": round(doc["score"], 3),
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "relevance": relevance_percent
        }

    def query_with_history(self, history: List[Dict], user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Поиск + генерация ответа с учётом истории диалога

        Args:
            history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            user_query: Текущий запрос
            top_k: Количество релевантных фрагментов

        Returns:
            {"answer": "ответ", "sources": [...], "relevance": float}
        """
        logger.info(f"💬 Запрос с историей: {user_query}")

        # 1. Поиск релевантных фрагментов
        documents = self.vector_store.search(user_query, top_k=top_k)

        if not documents:
            return {
                "answer": "❌ Не удалось найти подходящие документы.",
                "sources": [],
                "relevance": 0.0
            }

        # Вычисляем среднюю релевантность
        avg_score = sum(doc.get('score', 0) for doc in documents) / len(documents)
        relevance_percent = avg_score * 100

        # 2. Формируем контекст из документов
        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[Источник {idx}: {doc['file']}, стр. {doc['page']}]\n{doc['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # 3. Формируем историю диалога
        history_text = ""
        if history:
            history_lines = []
            for msg in history[-10:]:  # Берём последние 10 сообщений
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    prefix = "Пользователь:" if role == "user" else "Ассистент:"
                    history_lines.append(f"{prefix} {content}")

            if history_lines:
                history_text = "\n".join(history_lines)

        # 4. Генерируем ответ через LLM
        system_prompt = """Ты — AI-ассистент для работы с технической документацией по лифтам и лифтовому оборудованию.

        ТВОИ ЖЁСТКИЕ ПРАВИЛА:
        - Отвечай ТОЛЬКО на основе переданного контекста документов.
        - НИКОГДА не используй общие знания, интернет или свои догадки.
        - Если точной информации в контексте НЕТ — прямо так и скажи: 
          "В предоставленных фрагментах документации нет точной информации по этому вопросу."
        - НЕ подменяй одно устройство/плату другим.
        - Отвечай кратко и по делу, на русском языке.
        - Если есть таблица или явная инструкция — приведи их словами или структурированно.

        КРИТИЧЕСКИ ВАЖНО:
        - НЕ ПРИДУМЫВАЙ и НЕ ДОГАДЫВАЙСЯ.
        - Лучше честно ответь "Документация по <термин> в найденных фрагментах не представлена", чем написать чушь."""

        prompt_parts = []

        if history_text:
            prompt_parts.append(f"История диалога:\n{history_text}\n")

        prompt_parts.append(f"Контекст из документации:\n{context}\n")
        prompt_parts.append(f"Текущий вопрос пользователя:\n{user_query}\n")
        prompt_parts.append("Ответ:")

        prompt = "\n".join(prompt_parts)

        answer = self.ollama.generate(prompt, system_prompt=system_prompt)

        # 5. Формируем источники
        sources = [
            {
                "file": doc["file"],
                "page": doc["page"],
                "score": round(doc["score"], 3),
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "relevance": relevance_percent
        }

    def generate_clarification_questions(self, user_query: str) -> List[str]:
        """Генерация уточняющих вопросов"""
        system_prompt = """Ты помощник технической поддержки.
Пользователь задал неоднозначный вопрос.
Сгенерируй 3-5 коротких уточняющих вопросов (каждый на отдельной строке).
Вопросы должны быть ОЧЕНЬ короткими (максимум 5-7 слов)."""

        prompt = f"""Вопрос пользователя: {user_query}

Уточняющие вопросы:"""

        try:
            response = self.ollama.generate(prompt, system_prompt=system_prompt)

            # Парсим вопросы
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            questions = [q.lstrip('0123456789.-) ') for q in questions]  # Убираем нумерацию

            return questions[:5]  # Максимум 5 вопросов

        except Exception as e:
            logger.error(f"❌ Ошибка генерации уточняющих вопросов: {repr(e)}")
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

        # 1. Тест Ollama
        try:
            ok = self.ollama.test_connection()
            if ok:
                results.append(f"✅ Ollama: подключение успешно, модель {self.ollama.model} доступна.")
            else:
                results.append(f"❌ Ollama: не удалось подтвердить доступность модели {self.ollama.model}.")
        except Exception as e:
            results.append(f"❌ Ollama: ошибка подключения: {repr(e)}")

        # 2. Тест Qdrant
        try:
            qdrant_test = self.vector_store.test_connection()
            if isinstance(qdrant_test, dict) and "message" in qdrant_test:
                results.append(qdrant_test["message"])
            else:
                results.append(f"ℹ️ Qdrant: {qdrant_test}")
        except Exception as e:
            results.append(f"❌ Qdrant: ошибка подключения: {repr(e)}")

        return {"message": "\n\n".join(results)}