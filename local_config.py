import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем local.env
load_dotenv("local.env")

BASE_DIR = Path(__file__).resolve().parent

# === Telegram ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# === Логирование ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# === Папки ===
DOCUMENTS_FOLDER = BASE_DIR / "documents"
SESSIONS_FOLDER = BASE_DIR / "sessions"
EMBEDDING_CACHE_FOLDER = BASE_DIR / "embedding_cache"

DOCUMENTS_FOLDER.mkdir(exist_ok=True)
SESSIONS_FOLDER.mkdir(exist_ok=True)
EMBEDDING_CACHE_FOLDER.mkdir(exist_ok=True)

# === Ollama ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))

# === Эмбеддинги ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

# === Qdrant ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tech_docs").strip()

# === RAG ===
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# === OCR / Таблицы ===
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
ENABLE_TABLES = os.getenv("ENABLE_TABLES", "true").lower() == "true"
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "ru").split(",")

# === Чистка текста через LLM ===
ENABLE_TEXT_CLEANING = os.getenv("ENABLE_TEXT_CLEANING", "false").lower() == "true"

# === Docling ===
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "false").lower() == "true"
MAX_DOCLING_PAGES = int(os.getenv("MAX_DOCLING_PAGES", "9999"))  # Убрано ограничение

# === История диалога ===
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "6000"))

# === Стартовые данные от пользователя ===
INITIAL_DATA_FIELDS = [
    "Номер контракта",
    "Телефон",
    "Модель лифта",
    "Скорость",
    "Количество остановок",
    "Грузоподъёмность",
    "Город",
]


def check_config() -> bool:
    ok = True

    if not TELEGRAM_BOT_TOKEN:
        print("❌ Не задан TELEGRAM_BOT_TOKEN в local.env")
        ok = False

    print("⚙️ Конфигурация:")
    print(f"   • Документы: {DOCUMENTS_FOLDER}")
    print(f"   • Сессии: {SESSIONS_FOLDER}")
    print(f"   • Кэш эмбеддингов: {EMBEDDING_CACHE_FOLDER}")
    print(f"   • Ollama: {OLLAMA_BASE_URL}, модель: {OLLAMA_MODEL}")
    print(f"   • Qdrant: {QDRANT_URL}, коллекция: {QDRANT_COLLECTION}")
    print(f"   • Эмбеддинги: {EMBEDDING_MODEL}")
    print(f"   • OCR: {'включен' if ENABLE_OCR else 'выключен'}")
    print(f"   • Таблицы: {'включены' if ENABLE_TABLES else 'выключены'}")
    print(f"   • Чистка текста LLM: {'включена' if ENABLE_TEXT_CLEANING else 'выключена'}")
    print(f"   • Docling: {'включён' if ENABLE_DOCLING else 'выключен'}")
    print(f"   • История диалога: до {MAX_HISTORY_CHARS} символов")
    print(f"   • Размер чанка: {CHUNK_SIZE} символов, перекрытие: {CHUNK_OVERLAP}")

    return ok