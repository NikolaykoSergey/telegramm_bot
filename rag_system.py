import logging
from typing import List, Dict
from pathlib import Path

from document_processor import DocumentProcessor
from vector_store import VectorStore
from chatllm_client import ChatLLMClient
from config import DOCUMENTS_FOLDER, TOP_K_RESULTS, ENABLE_CHATLLM

logger = logging.getLogger(__name__)


class RAGSystem:
    """Система RAG для поиска и генерации ответов"""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.chatllm_client = ChatLLMClient()
        self.indexing_in_progress = False
        self.stop_indexing = False
        logger.info("✅ RAG система инициализирована")

    def index_documents(self, folder_path: Path = DOCUMENTS_FOLDER, save_every: int = 1,
                        continue_indexing: bool = True):
        """
        Индексация документов с возможностью остановки и продолжения

        Args:
            folder_path: Путь к папке с документами
            save_every: Как часто сохранять индекс (в файлах)
            continue_indexing: Продолжить с места остановки или начать заново
        """
        if self.indexing_in_progress:
            logger.warning("⚠️ Индексация уже выполняется")
            return

        self.indexing_in_progress = True
        self.stop_indexing = False

        logger.info("🔄 Начало индексации документов из %s", folder_path)

        if not folder_path.exists():
            logger.warning("⚠️ Папка %s не существует, создаю...", folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)
            self.indexing_in_progress = False
            return

        # Получаем список всех файлов
        all_files = [
            f for f in sorted(folder_path.rglob('*'))
            if f.is_file() and f.suffix.lower() in ('.pdf', '.docx', '.doc')
        ]

        if not all_files:
            logger.warning("⚠️ Не найдено ни одного PDF/DOC/DOCX файла для индексации")
            self.indexing_in_progress = False
            return

        # Загружаем прогресс
        indexed_files = []
        if continue_indexing:
            indexed_files = self.vector_store.load_progress()
            if indexed_files:
                logger.info("📂 Найдено %d уже проиндексированных файлов", len(indexed_files))
                # Загружаем существующий индекс
                self.vector_store.load()
        else:
            # Полная переиндексация - очищаем всё
            self.vector_store.clear()
            self.vector_store.clear_progress()

        # Фильтруем файлы, которые уже проиндексированы
        files_to_process = [f for f in all_files if f.name not in indexed_files]

        if not files_to_process:
            logger.info("✅ Все файлы уже проиндексированы")
            self.indexing_in_progress = False
            return

        logger.info("📄 Файлов к обработке: %d из %d", len(files_to_process), len(all_files))

        processed_files = 0
        total_fragments = 0

        for file_idx, file_path in enumerate(files_to_process, start=1):
            # Проверка флага остановки
            if self.stop_indexing:
                logger.warning("⚠️ Индексация остановлена пользователем")
                break

            logger.info("📄 [%d/%d] Обработка файла: %s", file_idx, len(files_to_process), file_path.name)

            try:
                content_list = self.document_processor.process_file(file_path)

                # Добавляем метаданные о файле
                for content in content_list:
                    content['file'] = file_path.name

                if not content_list:
                    logger.warning("⚠️ В файле %s не найдено текста/таблиц/OCR", file_path.name)
                    # Всё равно добавляем в список проиндексированных
                    indexed_files.append(file_path.name)
                    continue

                # Добавляем фрагменты в индекс
                self.vector_store.add_documents(content_list)

                processed_files += 1
                total_fragments += len(content_list)
                indexed_files.append(file_path.name)

                logger.info("   ✅ Файл %s: %d фрагментов добавлено", file_path.name, len(content_list))

                # Периодическое сохранение
                if processed_files % save_every == 0:
                    logger.info("💾 Промежуточное сохранение индекса после %d файлов...", processed_files)
                    self.vector_store.save()
                    self.vector_store.save_progress(indexed_files)

            except Exception as e:
                logger.error("❌ Ошибка при обработке файла %s: %s", file_path.name, repr(e))
                # Продолжаем со следующим файлом

        # Финальное сохранение
        logger.info("💾 Финальное сохранение индекса...")
        self.vector_store.save()
        self.vector_store.save_progress(indexed_files)

        if self.stop_indexing:
            logger.info(
                "⏸️ Индексация приостановлена: файлов=%d/%d, фрагментов=%d",
                len(indexed_files),
                len(all_files),
                total_fragments,
            )
        else:
            logger.info(
                "✅ Индексация завершена: файлов=%d, фрагментов=%d",
                len(indexed_files),
                total_fragments,
            )

        self.indexing_in_progress = False
        self.stop_indexing = False

    def stop_indexing_process(self):
        """Остановка процесса индексации"""
        if self.indexing_in_progress:
            logger.info("🛑 Запрос на остановку индексации...")
            self.stop_indexing = True
        else:
            logger.warning("⚠️ Индексация не выполняется")

    def is_indexing(self) -> bool:
        """Проверка, выполняется ли индексация"""
        return self.indexing_in_progress

    def load_index(self) -> bool:
        """Загрузка существующего индекса"""
        return self.vector_store.load()

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Поиск документов по запросу"""
        return self.vector_store.search(query, top_k)

    def query(self, user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """Полный цикл RAG: поиск + генерация ответа"""
        logger.info("🤖 RAG запрос: %s", user_query)

        search_results = self.search(user_query, top_k)

        if not search_results:
            return {
                'answer': (
                    "❌ В индексе нет релевантной информации или индекс пуст.\n\n"
                    "💡 Попробуйте:\n"
                    "• Выполнить /reindex для переиндексации\n"
                    "• Проверить, что документы лежат в папке `documents/`"
                ),
                'sources': [],
            }

        if not ENABLE_CHATLLM:
            # Только локальный поиск
            logger.warning("⚠️ ChatLLM отключен, RAG-ответы не доступны")
            return {
                'answer': (
                    "⚠️ AI-режим отключён (ENABLE_CHATLLM=False).\n\n"
                    "Доступен только локальный поиск `/search`.\n"
                    "Релевантные фрагменты найдены, но ответ не сформирован моделью."
                ),
                'sources': [
                    {
                        'file': doc.get('file', 'Unknown'),
                        'page': doc.get('page', 'N/A'),
                        'type': doc.get('type', 'text'),
                        'score': doc.get('score', 0.0),
                    }
                    for doc in search_results
                ],
            }

        answer = self.chatllm_client.generate_response(user_query, search_results)

        sources = []
        for doc in search_results:
            sources.append({
                'file': doc.get('file', 'Unknown'),
                'page': doc.get('page', 'N/A'),
                'type': doc.get('type', 'text'),
                'score': doc.get('score', 0.0),
            })

        return {
            'answer': answer,
            'sources': sources,
        }

    def query_with_history(self, history: List[Dict], user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Полный цикл RAG с учётом истории диалога: поиск + генерация ответа.

        Args:
            history: История диалога [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            user_query: Текущий запрос пользователя
            top_k: Количество релевантных фрагментов

        Returns:
            Dict с ключами 'answer' и 'sources'
        """
        logger.info("🤖 RAG запрос с историей: %s", user_query)

        search_results = self.search(user_query, top_k)

        if not search_results:
            return {
                'answer': (
                    "❌ В индексе нет релевантной информации или индекс пуст.\n\n"
                    "💡 Попробуйте:\n"
                    "• Выполнить /reindex для переиндексации\n"
                    "• Проверить, что документы лежат в папке `documents/`"
                ),
                'sources': [],
            }

        if not ENABLE_CHATLLM:
            logger.warning("⚠️ ChatLLM отключен, RAG-ответы не доступны")
            return {
                'answer': (
                    "⚠️ AI-режим отключён (ENABLE_CHATLLM=False).\n\n"
                    "Доступен только локальный поиск `/search`.\n"
                    "Релевантные фрагменты найдены, но ответ не сформирован моделью."
                ),
                'sources': [
                    {
                        'file': doc.get('file', 'Unknown'),
                        'page': doc.get('page', 'N/A'),
                        'type': doc.get('type', 'text'),
                        'score': doc.get('score', 0.0),
                    }
                    for doc in search_results
                ],
            }

        # Генерируем ответ с учётом истории
        answer = self.chatllm_client.generate_response_with_history(history, user_query, search_results)

        sources = []
        for doc in search_results:
            sources.append({
                'file': doc.get('file', 'Unknown'),
                'page': doc.get('page', 'N/A'),
                'type': doc.get('type', 'text'),
                'score': doc.get('score', 0.0),
            })

        return {
            'answer': answer,
            'sources': sources,
        }



    def generate_clarification_questions(self, user_query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """
        Генерирует уточняющие вопросы для неоднозначного запроса пользователя.
        Возвращает список коротких вопросов.
        """
        logger.info("❓ Генерация уточняющих вопросов для: %s", user_query)

        # Получаем контекст для генерации вопросов
        search_results = self.search(user_query, top_k)

        if not search_results:
            logger.warning("⚠️ Нет контекста для генерации вопросов")
            return []

        # Генерируем вопросы через ChatLLM
        questions = self.chatllm_client.generate_clarification_questions(user_query, search_results)

        return questions

    def get_stats(self) -> Dict:
        """Статистика"""
        stats = self.vector_store.get_stats()

        # Добавляем информацию о прогрессе индексации
        indexed_files = self.vector_store.load_progress()
        stats['indexed_files_count'] = len(indexed_files)
        stats['indexed_files_list'] = indexed_files

        return stats

    def test_connection(self) -> Dict:
        """Полное тестирование с диагностикой"""
        return self.chatllm_client.test_connection()