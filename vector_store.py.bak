import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config import INDEX_FILE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStore:
    """Векторное хранилище с FAISS"""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.progress_file = Path("indexed_files.json")
        logger.info("✅ VectorStore инициализирован (размерность: %d)", self.dimension)

    def add_documents(self, docs: List[Dict]):
        """Добавляет документы в индекс"""
        if not docs:
            return

        texts = [doc['content'] for doc in docs]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings.astype('float32'))
        self.documents.extend(docs)

        logger.debug("Добавлено %d документов в индекс", len(docs))

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск по запросу"""
        if self.index.ntotal == 0:
            logger.warning("Индекс пуст")
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(1 / (1 + dist))
                results.append(doc)

        return results

    def save(self):
        """Сохранение индекса"""
        try:
            faiss.write_index(self.index, str(INDEX_FILE))
            with open(str(INDEX_FILE) + '.docs', 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info("💾 Индекс сохранён: %s", INDEX_FILE)
        except Exception as e:
            logger.error("❌ Ошибка при сохранении индекса: %s", repr(e))

    def load(self) -> bool:
        """Загрузка индекса"""
        try:
            if not INDEX_FILE.exists():
                logger.warning("Файл индекса не найден: %s", INDEX_FILE)
                return False

            self.index = faiss.read_index(str(INDEX_FILE))
            with open(str(INDEX_FILE) + '.docs', 'rb') as f:
                self.documents = pickle.load(f)

            logger.info("✅ Индекс загружен: %d документов", len(self.documents))
            return True
        except Exception as e:
            logger.error("❌ Ошибка при загрузке индекса: %s", repr(e))
            return False

    def clear(self):
        """Полная очистка индекса"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        logger.info("🗑️ Индекс очищен")

    def get_stats(self) -> Dict:
        """Статистика по индексу"""
        files = list(set(doc.get('file', 'Unknown') for doc in self.documents))
        types = Counter(doc.get('type', 'unknown') for doc in self.documents)

        return {
            'total_documents': len(self.documents),
            'total_files': len(files),
            'files': sorted(files),
            'types': dict(types),
        }

    def save_progress(self, indexed_files: List[str]):
        """Сохраняет список проиндексированных файлов"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump({'indexed_files': indexed_files}, f, ensure_ascii=False, indent=2)
            logger.info("💾 Прогресс сохранён: %d файлов", len(indexed_files))
        except Exception as e:
            logger.error("❌ Ошибка при сохранении прогресса: %s", repr(e))

    def load_progress(self) -> List[str]:
        """Загружает список проиндексированных файлов"""
        try:
            if not self.progress_file.exists():
                return []

            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                indexed_files = data.get('indexed_files', [])
                logger.info("✅ Прогресс загружен: %d файлов", len(indexed_files))
                return indexed_files
        except Exception as e:
            logger.error("❌ Ошибка при загрузке прогресса: %s", repr(e))
            return []

    def clear_progress(self):
        """Удаляет файл прогресса"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                logger.info("🗑️ Файл прогресса удалён")
        except Exception as e:
            logger.error("❌ Ошибка при удалении прогресса: %s", repr(e))