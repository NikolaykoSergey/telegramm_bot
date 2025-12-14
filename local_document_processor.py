import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import pdfplumber
from tqdm import tqdm

from local_config import (
    ENABLE_OCR,
    ENABLE_TABLES,
    OCR_LANGUAGES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ENABLE_TEXT_CLEANING,
    ENABLE_DOCLING,
    MAX_DOCLING_PAGES,
)

from local_ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Пытаемся подключить PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("⚠️ PaddleOCR не установлен, OCR будет отключён.")

# Пытаемся подключить Docling
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("⚠️ Docling не установлен, функционал Docling будет отключён.")


def is_trash_text(text: str) -> bool:
    """
    Проверка, годится ли текст для индексации.
    Возвращает True, если текст — мусор (слишком мало, битая кодировка, артефакты).
    """
    if not text:
        return True

    cleaned = text.strip()
    if len(cleaned) < 200:  # менее 200 символов — мало для мануала
        return True

    # Доля нормальных букв vs остального
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned)
    ratio = len(letters) / max(len(cleaned), 1)
    if ratio < 0.3:
        # меньше 30% букв — похоже на мусор
        return True

    return False


class TextCleaner:
    """Чистка текста через LLM (Ollama)"""

    def __init__(self):
        self.enabled = ENABLE_TEXT_CLEANING
        self.ollama = OllamaClient()

    def clean_text(self, text: str, file_name: str = "", page: int = 0) -> str:
        if not self.enabled or not text.strip():
            return text

        system_prompt = (
            "Ты помощник, который очищает текст технической документации.\n\n"
            "ЗАДАЧА:\n"
            "- Удали повторы строк, мусор, обрезанные фрагменты.\n"
            "- Сохрани технические обозначения, ГОСТы, номера схем и т.п.\n"
            "- Не сокращай смысл, не перефразируй сильно.\n"
            "- Просто сделай текст аккуратным для дальнейшей индексации."
        )

        user_prompt = f"Файл: {file_name}, страница: {page}\n\nТекст:\n{text}"

        try:
            cleaned = self.ollama.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=512,
            )
            return cleaned.strip()
        except Exception as e:
            logger.error(f"❌ Ошибка чистки текста через LLM: {repr(e)}")
            return text


class DocumentProcessor:
    """Обработка документов: PDF, DOCX, извлечение текста, OCR, Docling, чанки"""

    def __init__(self):
        self.text_cleaner = TextCleaner()

        # OCR
        if ENABLE_OCR and PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='ru',
                    use_gpu=False,
                    show_log=False,
                )
                logger.info("✅ PaddleOCR инициализирован")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации PaddleOCR: {repr(e)}")
                self.ocr = None
        else:
            self.ocr = None
            if ENABLE_OCR and not PADDLEOCR_AVAILABLE:
                logger.warning("⚠️ ENABLE_OCR=true, но PaddleOCR не установлен")

        # Docling
        self.use_docling = ENABLE_DOCLING and DOCLING_AVAILABLE
        if self.use_docling:
            try:
                self.docling_converter = DocumentConverter()
                logger.info("✅ Docling инициализирован")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации Docling: {repr(e)}")
                self.use_docling = False
                self.docling_converter = None
        else:
            if ENABLE_DOCLING and not DOCLING_AVAILABLE:
                logger.warning("⚠️ ENABLE_DOCLING=true, но Docling не установлен (pip install docling)")
            self.docling_converter = None

        logger.info("✅ DocumentProcessor инициализирован (OCR=%s, Docling=%s)", bool(self.ocr), self.use_docling)

    def process_file(self, file_path: Path) -> List[Dict]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext == ".docx":
            return self._process_docx(file_path)
        else:
            logger.warning(f"⚠️ Неподдерживаемый формат: {file_path.name}")
            return []

    def _process_pdf(self, file_path: Path) -> List[Dict]:
        """
        Робастная обработка PDF по схеме:
        1) pdfplumber (текст + таблицы)
        2) если мало текста → docling
        3) если и docling не дал → OCR по картинкам страниц
        4) чистка через LLM
        5) чанки → векторизация
        """
        fragments = []

        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"📄 PDF: {file_path.name}, страниц: {num_pages}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.debug(f"   Обработка страницы {page_num}/{num_pages}...")

                    # 1. БАЗОВЫЙ ТЕКСТ (pdfplumber)
                    text = page.extract_text() or ""
                    text = text.strip()

                    # 2. ТАБЛИЦЫ
                    tables_text = ""
                    if ENABLE_TABLES:
                        tables = page.extract_tables()
                        if tables:
                            tables_text = self._format_tables(tables)
                            logger.debug(f"      ✅ Найдено таблиц: {len(tables)}")

                    # Объединяем базовый текст + таблицы
                    combined_text = "\n\n".join(part for part in [text, tables_text] if part and part.strip()).strip()

                    # 3. ПРОВЕРКА: если текста мало или мусор → пробуем docling
                    if is_trash_text(combined_text):
                        logger.debug(f"      ⚠️ Мало текста ({len(combined_text)} симв.), пробуем docling...")

                        docling_text = self._extract_page_with_docling(file_path, page_num)
                        if docling_text and not is_trash_text(docling_text):
                            logger.debug(f"      ✅ Docling дал {len(docling_text)} символов")
                            combined_text = docling_text
                        else:
                            logger.debug(f"      ⚠️ Docling тоже не помог, идём в OCR...")

                            # 4. OCR ПО КАРТИНКЕ СТРАНИЦЫ
                            if self.ocr:
                                ocr_text = self._ocr_page_image(page)
                                if ocr_text and not is_trash_text(ocr_text):
                                    logger.debug(f"      ✅ OCR дал {len(ocr_text)} символов")
                                    combined_text = ocr_text
                                else:
                                    logger.debug(f"      ❌ OCR тоже не дал нормального текста")

                    # Если после всех попыток текста нет — пропускаем страницу
                    if not combined_text or is_trash_text(combined_text):
                        logger.debug(f"      ⚠️ Страница {page_num} пустая после всех попыток, пропускаю")
                        continue

                    # 5. ЧИСТКА ЧЕРЕЗ LLM
                    cleaned_text = self.text_cleaner.clean_text(
                        combined_text,
                        file_name=file_path.name,
                        page=page_num,
                    )

                    # 6. ЧАНКИ
                    chunks = self._split_into_chunks(cleaned_text)

                    for chunk in chunks:
                        fragments.append({
                            "content": chunk,
                            "page": page_num,
                            "type": "text",
                            "file": file_path.name,
                        })

                logger.info(f"✅ PDF {file_path.name}: извлечено {len(fragments)} фрагментов")
                return fragments

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке PDF {file_path.name}: {repr(e)}")
            return []

    def _extract_page_with_docling(self, file_path: Path, page_num: int) -> str:
        """
        Извлечение текста одной страницы через Docling.
        Возвращает текст страницы или пустую строку.
        """
        if not self.use_docling or not self.docling_converter:
            return ""

        if page_num > MAX_DOCLING_PAGES:
            return ""

        try:
            result = self.docling_converter.convert(str(file_path))

            for page in result.document.pages:
                if page.page_number == page_num:
                    lines = []
                    for block in page.blocks:
                        txt = block.to_text().strip()
                        if txt:
                            lines.append(txt)

                    if lines:
                        return "\n".join(lines)

            return ""

        except Exception as e:
            logger.error(f"❌ Ошибка Docling при обработке страницы {page_num} файла {file_path.name}: {repr(e)}")
            return ""

    def _ocr_page_image(self, page) -> str:
        """OCR страницы через PaddleOCR (из pdfplumber page)"""
        if not self.ocr:
            return ""

        try:
            # Конвертируем страницу в изображение
            logger.debug("    🖼 Преобразование страницы в изображение...")
            img = page.to_image(resolution=150).original

            logger.debug("    🔠 Запуск PaddleOCR...")
            result = self.ocr.ocr(img, cls=True)

            if not result or not result[0]:
                logger.debug("    ⚠️ OCR не нашёл текста на странице")
                return ""

            lines = []
            for line in result[0]:
                text = line[1][0] if len(line) > 1 else ""
                if text:
                    lines.append(text)

            ocr_text = "\n".join(lines)
            logger.debug(f"    ✅ OCR извлёк {len(ocr_text)} символов")
            return ocr_text

        except AssertionError as e:
            logger.error(f"❌ AssertionError в OCR: {repr(e)}")
            return ""
        except Exception as e:
            logger.error(f"❌ Ошибка OCR: {repr(e)}")
            return ""

    def _format_tables(self, tables) -> str:
        """Форматирование таблиц в текст"""
        parts = []
        for t in tables:
            for row in t:
                row = [str(cell).strip() if cell else "" for cell in row]
                parts.append(" | ".join(row))
            parts.append("\n")
        return "\n".join(parts)

    def _process_docx(self, file_path: Path) -> List[Dict]:
        """Простейшая обработка DOCX"""
        try:
            from docx import Document
        except ImportError:
            logger.error("❌ Для обработки DOCX нужен пакет python-docx (pip install python-docx)")
            return []

        fragments = []
        try:
            doc = Document(str(file_path))
            full_text = []

            for para in doc.paragraphs:
                txt = (para.text or "").strip()
                if txt:
                    full_text.append(txt)

            combined = "\n".join(full_text).strip()
            if not combined:
                return []

            cleaned_text = self.text_cleaner.clean_text(
                combined,
                file_name=file_path.name,
                page=1,
            )

            chunks = self._split_into_chunks(cleaned_text)
            for chunk in chunks:
                fragments.append({
                    "content": chunk,
                    "page": 1,
                    "type": "text",
                    "file": file_path.name,
                })

            logger.info(f"✅ DOCX {file_path.name}: извлечено {len(fragments)} фрагментов")
            return fragments

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке DOCX {file_path.name}: {repr(e)}")
            return []

    def _split_into_chunks(self, text: str) -> List[str]:
        """Режем текст на чанки"""
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + CHUNK_SIZE, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks