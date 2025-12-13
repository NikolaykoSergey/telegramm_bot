import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из текущей директории (где лежит config.py) или выше
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # Попробуем глобально (на случай, если .env в корне проекта и запуск из другого места)
    load_dotenv()

# === БАЗОВЫЕ НАСТРОЙКИ ===

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Документы
DOCUMENTS_FOLDER = BASE_DIR / "documents"

# RAG / эмбеддинги / FAISS
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
INDEX_FILE = BASE_DIR / "vector_index.pkl"
FAISS_INDEX_FILE = BASE_DIR / "faiss.index"
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Чанки
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# OCR / PDF парсинг
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "rus+eng")
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
ENABLE_TABLES = os.getenv("ENABLE_TABLES", "true").lower() == "true"

# Если 0 или меньше — значит "все страницы"
MAX_PDF_PAGES_FOR_OCR = int(os.getenv("MAX_PDF_PAGES_FOR_OCR", "0"))
MAX_PDF_PAGES_FOR_TABLES = os.getenv("MAX_PDF_PAGES_FOR_TABLES", "all")  # 'all' или '1-5'

# ChatLLM API
CHATLLM_API_KEY = os.getenv("CHATLLM_API_KEY", "").strip()
CHATLLM_API_URL = os.getenv(
    "CHATLLM_API_URL",
    "https://routellm.abacus.ai/route",  # дефолт под RouteLLM
).strip()
# Изменено на Gemini 1.5 Pro для оптимального баланса цена/качество
CHATLLM_MODEL = os.getenv("CHATLLM_MODEL", "").strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
ENABLE_CHATLLM = os.getenv("ENABLE_CHATLLM", "true").lower() == "true"


def check_config() -> bool:
    ok = True

    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN не задан (см. .env)")
        ok = False

    if ENABLE_CHATLLM and not CHATLLM_API_KEY:
        print("⚠️ ENABLE_CHATLLM=True, но CHATLLM_API_KEY не задан. AI-режим работать не будет.")

    return ok