import logging
from typing import List, Dict
import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from local_config import QDRANT_URL, QDRANT_COLLECTION
from local_embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """Векторное хранилище на базе Qdrant с улучшенным EmbeddingManager"""

    def __init__(self, embedding_model: str = None):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

        # Используем EmbeddingManager
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.vector_size = self.embedding_manager.get_embedding_dimension()

        self._ensure_collection()

        logger.info(f"✅ VectorStore инициализирован: {QDRANT_URL}")
        logger.info(f"   📊 Коллекция: {self.collection_name}")
        logger.info(f"   🔤 Модель: {self.embedding_manager.model_name}")
        logger.info(f"   📐 Размерность: {self.vector_size}")

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

                # Проверяем размерность
                info = self.client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                if existing_size != self.vector_size:
                    logger.warning(f"⚠️ Размерность не совпадает: БД={existing_size}, модель={self.vector_size}")
                    logger.warning("   Требуется переиндексация с /reindex")

        except Exception as e:
            logger.error(f"❌ Ошибка при проверке/создании коллекции: {repr(e)}")
            raise

    def add_documents(self, documents: List[Dict]):
        """Добавление документов в векторное хранилище с детальным логированием"""
        if not documents:
            return

        logger.info(f"📤 Начинаю загрузку {len(documents)} документов в Qdrant...")
        start_time = time.time()

        try:
            points = []
            texts_to_encode = []
            indices_to_encode = []

            # Подготовка данных для батч-кодирования
            logger.info("📝 Подготовка текстов для кодирования...")
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                if not content:
                    continue

                texts_to_encode.append(content)
                indices_to_encode.append(i)

                # Логируем прогресс подготовки
                if (i + 1) % 50 == 0:
                    logger.info(f"📝 Подготовлено {i + 1}/{len(documents)} текстов...")

            logger.info(f"🔤 Начинаю кодирование {len(texts_to_encode)} текстов...")

            # Батч-кодирование с прогрессом
            embeddings = self.embedding_manager.encode(texts_to_encode, batch_size=32)

            logger.info("🎯 Создание векторных точек...")
            # Создание точек
            for idx, doc_idx in enumerate(indices_to_encode):
                doc = documents[doc_idx]
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx].tolist(),
                    payload={
                        "content": doc.get("content", ""),
                        "file": doc.get("file", ""),
                        "page": doc.get("page", 0),
                        "type": doc.get("type", "text"),
                    },
                )
                points.append(point)

                # Логируем прогресс создания точек
                if (idx + 1) % 100 == 0:
                    logger.info(f"🎯 Создано {idx + 1}/{len(indices_to_encode)} точек...")

            if points:
                # Загрузка в Qdrant
                logger.info(f"📤 Загружаю {len(points)} точек в Qdrant...")
                upload_start = time.time()

                # Загружаем батчами по 100 точек
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
                    percent = min(100, (i + len(batch)) / len(points) * 100)
                    logger.info(f"📤 Загружено {min(i + len(batch), len(points))}/{len(points)} точек ({percent:.1f}%)")

                upload_time = time.time() - upload_start

                total_time = time.time() - start_time
                logger.info(f"✅ Успешно добавлено {len(points)} документов в Qdrant")
                logger.info(f"   ⏱️ Общее время: {total_time:.1f}с")
                logger.info(f"   ⏱️ Время кодирования: {total_time - upload_time:.1f}с")
                logger.info(f"   ⏱️ Время загрузки: {upload_time:.1f}с")
                logger.info(f"   📈 Скорость: {len(points) / total_time:.1f} док/сек")

        except Exception as e:
            logger.error(f"❌ Критическая ошибка при добавлении документов в Qdrant: {repr(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск релевантных документов"""
        logger.debug(f"🔍 Начинаю поиск: '{query[:50]}...', top_k={top_k}")
        search_start = time.time()

        try:
            # Генерируем эмбеддинг запроса
            logger.debug("🔤 Кодирование запроса...")
            query_embedding = self.embedding_manager.encode(query, use_cache=False)[0].tolist()

            # Поиск в Qdrant
            logger.debug(f"🔎 Поиск в коллекции {self.collection_name}...")
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

            search_time = time.time() - search_start
            logger.info(f"✅ Найдено {len(documents)} документов за {search_time * 1000:.0f}мс")

            return documents

        except Exception as e:
            logger.error(f"❌ Ошибка при поиске в Qdrant: {repr(e)}")
            return []

    def clear_collection(self):
        """Очистка коллекции"""
        logger.warning("🗑️ Начинаю очистку векторной коллекции...")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            # Ждем немного
            import time
            time.sleep(1)
            self._ensure_collection()
            logger.info(f"✅ Коллекция {self.collection_name} успешно очищена и пересоздана")
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
                "model": self.embedding_manager.model_name,
            }
        except Exception as e:
            logger.error(f"❌ Ошибка при получении статистики: {repr(e)}")
            return {"total_documents": 0, "vector_size": 0, "model": "unknown"}

    def test_connection(self) -> Dict[str, str]:
        """Тест подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            msg = f"✅ Qdrant: подключение OK\n"
            msg += f"   📚 Коллекций: {len(collections.collections)}\n"
            msg += f"   🔤 Текущая коллекция: {self.collection_name} ({'существует' if self.collection_name in collection_names else 'не найдена'})\n"
            msg += f"   🧮 Модель эмбеддингов: {self.embedding_manager.model_name}\n"
            msg += f"   📐 Размерность: {self.vector_size}"

            return {
                "status": "ok",
                "message": msg,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Qdrant: ошибка подключения\n   {str(e)}\n   Проверь, запущен ли Qdrant (docker-compose up -d)",
            }