# local_vector_store_qdrant.py
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Callable

import requests
import numpy as np
from sentence_transformers import SentenceTransformer

from local_config import EMBEDDING_MODEL_NAME, QDRANT_URL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class LocalQdrantVectorStore:
    def __init__(self, model_name: Optional[str] = None):
        # берём модель из local_config, но допускаем override через аргумент
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        # явный fallback на MiniLM — будет использован при ошибке загрузки тяжелой модели
        self.fallback_model = "all-MiniLM-L6-v2"

        logger.info("🔤 Инициализация векторного стора. Запрошенная модель: %s", self.model_name)
        self.emb_model = self._load_embedding_model_with_fallback(self.model_name, self.fallback_model)

        self.qdrant_url = QDRANT_URL.rstrip("/")
        self.collection = QDRANT_COLLECTION

        self._ensure_collection()

    def _load_embedding_model_with_fallback(self, preferred: str, fallback: str) -> SentenceTransformer:
        """
        Пытается загрузить preferred модель. При любой ошибке — логирует и загружает fallback.
        Это делает работу стабильной на CPU и без геморроя с памятью.
        """
        try:
            # Если preferred явно содержит "bge" — принудительно не грузим на локальную машину,
            # чтобы не пытаться тащить тяжёлую модель в CPU среду.
            if preferred and "bge" in preferred.lower():
                raise RuntimeError("Preferred model looks like BGE/Gemma2 — skipping heavy local load.")

            logger.info("🔄 Пытаемся загрузить модель эмбеддингов: %s", preferred)
            model = SentenceTransformer(preferred, device="cpu", trust_remote_code=True)
            logger.info("✅ Успешно загружена модель: %s", preferred)
            return model
        except Exception as e:
            logger.warning("⚠️ Не удалось загрузить основную модель '%s': %s", preferred, repr(e))
            logger.info("🔁 Переключаемся на fallback-эмбеддер: %s", fallback)
            try:
                model = SentenceTransformer(fallback, device="cpu")
                logger.info("✅ Успешно загружен fallback: %s", fallback)
                # Обновим имя модели чтобы был понятный лог/индикация
                self.model_name = fallback
                return model
            except Exception as e2:
                logger.error("❌ Не удалось загрузить fallback-эмбеддер '%s': %s", fallback, repr(e2))
                raise RuntimeError("Failed to load embedding model and fallback") from e2

    def _ensure_collection(self):
        url = f"{self.qdrant_url}/collections/{self.collection}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("📚 Коллекция Qdrant '%s' уже существует", self.collection)
                return
        except Exception:
            # если ошибка — будем пытаться создать коллекцию
            pass

        logger.info("📚 Создаю коллекцию Qdrant '%s'...", self.collection)
        dim = self.emb_model.get_sentence_embedding_dimension()
        payload = {
            "vectors": {"size": dim, "distance": "Cosine"},
        }
        resp = requests.put(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("✅ Коллекция '%s' создана (размерность: %d)", self.collection, dim)

    def clear_collection(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Полное очищение коллекции (для полной переиндексации)."""
        if progress_callback:
            progress_callback({"stage": "clearing", "done": 0, "total": 1, "message": "Очищаем коллекцию Qdrant..."})
        url = f"{self.qdrant_url}/collections/{self.collection}/points/delete"
        try:
            resp = requests.post(url, json={"filter": {}}, timeout=60)
            resp.raise_for_status()
            if progress_callback:
                progress_callback({"stage": "clearing", "done": 1, "total": 1, "message": "Коллекция очищена."})
            logger.info("🗑 Коллекция очищена")
        except Exception as e:
            logger.warning("⚠️ Не удалось очистить коллекцию полностью: %s", e)
            # не фейлим, продолжим

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        h = self._hash_text(text)
        path = os.path.join(CACHE_DIR, f"{h}.npy")
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def _save_cached_embedding(self, text: str, emb: np.ndarray):
        h = self._hash_text(text)
        path = os.path.join(CACHE_DIR, f"{h}.npy")
        try:
            np.save(path, emb)
        except Exception as e:
            logger.warning("⚠️ Не удалось сохранить кэш эмбеддинга: %s", e)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch-encode with sentence-transformers (returns numpy array)."""
        embs = self.emb_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        return embs

    def rebuild_collection(self, docs: List[Dict[str, Any]], progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Загружает векторы в коллекцию (без удаления старых точек).
        Поддерживает progress_callback(progress_dict).
        progress_dict: {stage, done, total, message}
        """
        total_docs = len(docs)
        if total_docs == 0:
            logger.warning("⚠️ rebuild_collection: пустой список документов")
            return

        texts = [d["text"] for d in docs]

        # 1) Проверяем кэш
        vectors = [None] * total_docs
        uncached_texts = []
        uncached_indices = []
        for i, t in enumerate(texts):
            cached = self._get_cached_embedding(t)
            if cached is not None:
                vectors[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if progress_callback:
            progress_callback({"stage": "cache_check", "done": total_docs - len(uncached_texts), "total": total_docs, "message": "Проверка кэша эмбеддингов..."})

        # 2) Кодируем uncached партиями
        if uncached_texts:
            batch_enc = 32
            for start in range(0, len(uncached_texts), batch_enc):
                batch = uncached_texts[start:start + batch_enc]
                embs = self._encode_batch(batch)
                for j, emb in enumerate(embs):
                    idx = uncached_indices[start + j]
                    vectors[idx] = emb
                    self._save_cached_embedding(batch[j], emb)
                if progress_callback:
                    progress_callback({"stage": "encoding", "done": min(len(uncached_texts), start + batch_enc), "total": len(uncached_texts), "message": "Генерация эмбеддингов..."})
        else:
            if progress_callback:
                progress_callback({"stage": "encoding", "done": 0, "total": 0, "message": "Все эмбеддинги найдены в кэше."})

        # 3) Подготовка точек
        points = []
        for i, doc in enumerate(docs):
            vec = vectors[i]
            if vec is None:
                logger.warning("⚠️ Вектор для документа %d пустой — пропускаем", i)
                continue
            # ensure python floats
            vlist = np.array(vec).astype(float).tolist()
            points.append({
                "id": doc.get("id", i),
                "vector": vlist,
                "payload": {
                    "file": doc.get("file", ""),
                    "page": doc.get("page", 1),
                    "type": doc.get("type", "text"),
                    "text": doc.get("text", ""),
                },
            })

        # 4) Загрузка пачками
        upload_batch = 256
        url = f"{self.qdrant_url}/collections/{self.collection}/points?wait=true"
        for start in range(0, len(points), upload_batch):
            batch = points[start:start + upload_batch]
            try:
                resp = requests.post(url, json={"points": batch}, timeout=120)
                resp.raise_for_status()
            except Exception as e:
                logger.exception("❌ Ошибка загрузки батча в Qdrant: %s", e)
                # продолжаем попытки для следующих батчей
                continue
            if progress_callback:
                progress_callback({"stage": "upload", "done": min(len(points), start + upload_batch), "total": len(points), "message": "Загрузка векторов в Qdrant..."})

        if progress_callback:
            progress_callback({"stage": "done", "done": len(points), "total": len(points), "message": "Загрузка завершена."})

        logger.info("✅ В Qdrant загружено %d документов (прибл.)", len(points))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = self._encode_batch([query])[0].tolist()
        url = f"{self.qdrant_url}/collections/{self.collection}/points/search"
        payload = {"vector": vector, "limit": top_k, "with_payload": True, "with_vector": False}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = []
        for point in data.get("result", []):
            payload_data = point.get("payload", {}) or {}
            score = point.get("score", 0.0)
            result.append({
                "text": payload_data.get("text", ""),
                "file": payload_data.get("file", ""),
                "page": payload_data.get("page", 1),
                "type": payload_data.get("type", "text"),
                "score": score,
            })
        return result

    def stats(self) -> Dict[str, Any]:
        url = f"{self.qdrant_url}/collections/{self.collection}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            points_count = data.get("result", {}).get("points_count", 0)
        except Exception:
            points_count = 0
        return {"total_documents": points_count}
#