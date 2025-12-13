import logging
from typing import List, Dict
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from local_config import QDRANT_URL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)


class VectorStore:
    """Векторное хранилище на базе Qdrant"""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

        # Модель эмбеддингов
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()

        self._ensure_collection()

        logger.info(f"✅ VectorStore инициализирован: {QDRANT_URL}, коллекция: {self.collection_name}")

    def _ensure_collection(self):
        """Создание коллекции, если её нет"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"✅ Коллекция {self.collection_name} создана")
            else:
                logger.info(f"✅ Коллекция {self.collection_name} уже существует")

        except Exception as e:
            logger.error(f"❌ Ошибка при проверке/создании коллекции: {repr(e)}")
            raise

    def add_documents(self, documents: List[Dict]):
        """Добавление документов в векторное хранилище"""
        if not documents:
            return

        try:
            points = []

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                # Генерируем эмбеддинг
                embedding = self.embedding_model.encode(content).tolist()

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": content,
                        "file": doc.get("file", ""),
                        "page": doc.get("page", 0),
                        "type": doc.get("type", "text"),
                    },
                )
                points.append(point)

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"✅ Добавлено {len(points)} документов в Qdrant")

        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении документов: {repr(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск релевантных документов"""
        try:
            # Генерируем эмбеддинг запроса
            query_embedding = self.embedding_model.encode(query).tolist()

            # Поиск в Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )

            documents = []
            for result in results:
                documents.append({
                    "content": result.payload.get("content", ""),
                    "file": result.payload.get("file", ""),
                    "page": result.payload.get("page", 0),
                    "score": result.score,
                })

            return documents

        except Exception as e:
            logger.error(f"❌ Ошибка при поиске: {repr(e)}")
            return []

    def clear_collection(self):
        """Очистка коллекции"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"✅ Коллекция {self.collection_name} очищена")
        except Exception as e:
            logger.error(f"❌ Ошибка при очистке коллекции: {repr(e)}")
            raise

    def get_stats(self) -> Dict:
        """Статистика коллекции"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "total_documents": info.points_count,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as e:
            logger.error(f"❌ Ошибка при получении статистики: {repr(e)}")
            return {"total_documents": 0, "vector_size": 0}

    def test_connection(self) -> Dict[str, str]:
        """Тест подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            return {
                "status": "ok",
                "message": f"✅ Qdrant: подключение OK\n   Коллекций: {len(collections.collections)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Qdrant: ошибка подключения\n   {str(e)}\n   Проверь, запущен ли Qdrant (docker-compose up -d)",
            }
#