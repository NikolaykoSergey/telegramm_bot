import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid

from local_config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Работа с Qdrant (векторная БД)"""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

        # Загружаем модель эмбеддингов
        logger.info(f"📦 Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✅ Модель загружена (размерность: {VECTOR_SIZE})")

        # Создаём коллекцию если не существует
        self._ensure_collection()

    def _ensure_collection(self):
        """Создание коллекции если не существует"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"📁 Создаю коллекцию: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"✅ Коллекция {self.collection_name} создана")
            else:
                logger.info(f"✅ Коллекция {self.collection_name} уже существует")

        except Exception as e:
            logger.error(f"❌ Ошибка при создании коллекции: {repr(e)}")
            raise

    def add_documents(self, documents: List[Dict]):
        """
        Добавление документов в Qdrant

        Args:
            documents: [{"content": "текст", "page": 1, "file": "name.pdf", "type": "text"}, ...]
        """
        if not documents:
            logger.warning("⚠️ Нет документов для добавления")
            return

        logger.info(f"📥 Добавление {len(documents)} документов в Qdrant...")

        # Извлекаем тексты для эмбеддинга
        texts = [doc["content"] for doc in documents]

        # Генерируем эмбеддинги
        logger.info("🔄 Генерация эмбеддингов...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        )

        # Формируем точки для Qdrant
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),  # Уникальный ID
                vector=embedding.tolist(),
                payload={
                    "content": doc["content"],
                    "page": doc.get("page", 1),
                    "file": doc.get("file", "unknown"),
                    "type": doc.get("type", "text"),
                }
            )
            points.append(point)

        # Загружаем в Qdrant батчами
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(f"   ✅ Загружено {i + len(batch)}/{len(points)}")

        logger.info(f"✅ Добавлено {len(documents)} документов в Qdrant")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск похожих документов

        Args:
            query: Запрос пользователя
            top_k: Количество результатов

        Returns:
            [{"content": "текст", "page": 1, "file": "name.pdf", "score": 0.95}, ...]
        """
        logger.info(f"🔍 Поиск: '{query[:100]}...'")

        # Генерируем эмбеддинг запроса
        query_embedding = self.embedding_model.encode(query).tolist()

        # Поиск в Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        # Форматируем результаты
        documents = []
        for result in results:
            doc = {
                "content": result.payload.get("content", ""),
                "page": result.payload.get("page", 1),
                "file": result.payload.get("file", "unknown"),
                "type": result.payload.get("type", "text"),
                "score": result.score,
            }
            documents.append(doc)

        logger.info(f"✅ Найдено {len(documents)} документов")
        return documents

    def get_stats(self) -> Dict:
        """Статистика коллекции"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_documents": info.points_count,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {repr(e)}")
            return {"total_documents": 0, "vector_size": VECTOR_SIZE}

    def clear_collection(self):
        """Очистка коллекции"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ Коллекция {self.collection_name} удалена")
            self._ensure_collection()
        except Exception as e:
            logger.error(f"❌ Ошибка при очистке коллекции: {repr(e)}")

    def test_connection(self) -> Dict[str, str]:
        """Тест подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            msg = (
                f"✅ Подключение к Qdrant успешно!\n\n"
                f"• URL: {QDRANT_URL}\n"
                f"• Коллекция: {self.collection_name}\n"
                f"• Всего коллекций: {len(collections.collections)}"
            )
            logger.info(msg)
            return {"ok": "true", "message": msg}

        except Exception as e:
            msg = (
                f"❌ Не удалось подключиться к Qdrant.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n\n"
                f"Проверьте:\n"
                f"• Запущен ли Qdrant (docker-compose up -d)\n"
                f"• Правильность URL: {QDRANT_URL}"
            )
            logger.error(msg)
            return {"ok": "false", "message": msg}