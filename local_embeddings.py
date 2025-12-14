"""
Управление эмбеддингами с поддержкой BGE/Gemma
"""

import logging
import hashlib
import pickle
import os
from pathlib import Path
from typing import List, Union, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from local_config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Пытаемся импортировать FlagEmbedding
try:
    from FlagEmbedding import FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    logger.warning("⚠️ FlagEmbedding не установлен, используем только SentenceTransformer")


class EmbeddingCache:
    """Кэширование эмбеддингов на диск"""

    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"💾 Инициализирован кэш эмбеддингов: {self.cache_dir}")

    def get_hash(self, text: str) -> str:
        """Хеширование текста для имени файла"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга из кэша"""
        hash_id = self.get_hash(text)
        cache_file = self.cache_dir / f"{hash_id}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"💾 Кэш попадание: {hash_id[:8]}...")
                return embedding
            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения кэша {hash_id}: {repr(e)}")
                return None
        return None

    def set(self, text: str, embedding: np.ndarray):
        """Сохранение эмбеддинга в кэш"""
        hash_id = self.get_hash(text)
        cache_file = self.cache_dir / f"{hash_id}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"💾 Кэш сохранен: {hash_id[:8]}...")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения кэша {hash_id}: {repr(e)}")


class EmbeddingManager:
    """Менеджер эмбеддингов с поддержкой BGE/Gemma и кэшированием"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.cache = EmbeddingCache()
        self.model = None
        self.is_flag_model = False
        self._init_model()
        logger.info(f"🔤 EmbeddingManager инициализирован: {self.model_name}")

    def _init_model(self):
        """Инициализация модели эмбеддингов с fallback"""
        try:
            model_lower = self.model_name.lower()

            # Пробуем загрузить BGE через FlagEmbedding если доступно
            if "bge" in model_lower and FLAG_EMBEDDING_AVAILABLE:
                logger.info(f"🔄 Загружаем BGE модель через FlagEmbedding: {self.model_name}")

                # Выбираем инструкцию в зависимости от модели
                if "zh" in model_lower:
                    query_instruction = "为这个句子生成表示以用于检索相关文章："
                else:
                    query_instruction = "Represent this sentence for searching relevant passages:"

                self.model = FlagModel(
                    self.model_name,
                    query_instruction_for_retrieval=query_instruction,
                    use_fp16=False,  # Для CPU
                )
                self.is_flag_model = True
                logger.info(f"✅ BGE модель загружена: {self.model_name}")

            else:
                # Используем SentenceTransformer для остальных моделей
                logger.info(f"🔄 Загружаем SentenceTransformer: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device="cpu")
                self.is_flag_model = False
                logger.info(f"✅ SentenceTransformer загружен: {self.model_name}")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {self.model_name}: {repr(e)}")
            # Fallback на MiniLM
            logger.info("🔄 Загружаем fallback модель: all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            self.model_name = "all-MiniLM-L6-v2"
            self.is_flag_model = False

    def encode(self, texts: Union[str, List[str]], use_cache: bool = True, batch_size: int = 32, **kwargs) -> np.ndarray:
        """Кодирование текста в эмбеддинги с кэшированием"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Проверяем кэш
        if use_cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Кодируем не закэшированные тексты
        if uncached_texts:
            total_uncached = len(uncached_texts)
            logger.debug(f"🔤 Кодирование {total_uncached} текстов...")

            if self.is_flag_model:
                # FlagModel (BGE)
                if hasattr(self.model, 'encode_queries'):
                    # Для запросов
                    new_embeddings = self.model.encode_queries(uncached_texts)
                else:
                    # Для документов
                    new_embeddings = self.model.encode(uncached_texts)
            else:
                # SentenceTransformer - батч-обработка с логированием прогресса
                new_embeddings = []
                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[batch_start:batch_end]

                    # Логируем прогресс
                    percent = (batch_end / len(uncached_texts)) * 100
                    logger.debug(f"🔤 Кодирование батча: {batch_start+1}-{batch_end}/{len(uncached_texts)} ({percent:.1f}%)")

                    batch_embeddings = self.model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        **kwargs
                    )
                    new_embeddings.extend(batch_embeddings)

                new_embeddings = np.array(new_embeddings)

            # Сохраняем в кэш
            for i, (text, emb) in enumerate(zip(uncached_texts, new_embeddings)):
                self.cache.set(text, emb)

            # Собираем все эмбеддинги
            if use_cache:
                # Создаем полный список
                all_embeddings = [None] * len(texts)
                for i, emb in enumerate(embeddings):
                    all_embeddings[i] = emb

                for idx, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[idx] = emb

                return np.array(all_embeddings)
            else:
                return new_embeddings

        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """Получение размерности эмбеддингов"""
        # Пробуем определить размерность
        try:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()

            # Определяем по имени модели
            model_lower = self.model_name.lower()

            if "m3" in model_lower or "large" in model_lower:
                return 1024
            elif "base" in model_lower:
                return 768
            elif "mini" in model_lower:
                return 384
            elif "e5-large" in model_lower:
                return 1024
            elif "e5-base" in model_lower:
                return 768
            else:
                # Тестовая кодировка для определения размерности
                test_emb = self.encode("test", use_cache=False)
                return test_emb.shape[1]

        except Exception as e:
            logger.warning(f"⚠️ Не удалось определить размерность, используем 384: {repr(e)}")
            return 384

    def clear_cache(self):
        """Очистка кэша эмбеддингов"""
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        for file in cache_files:
            file.unlink()
        logger.info(f"🗑️ Очищен кэш эмбеддингов: {len(cache_files)} файлов")