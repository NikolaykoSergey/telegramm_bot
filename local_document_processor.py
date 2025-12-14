import logging
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
        """Гибридная обработка PDF (+ Docling, + OCR)"""
        fragments = []

        # Docling по всему документу (странично)
        docling_page_texts: Optional[Dict[int, str]] = None
        if self.use_docling:
            docling_page_texts = self._extract_with_docling(file_path)

        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"📄 PDF: {file_path.name}, страниц: {num_pages}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.debug(f"   Обработка страницы {page_num}/{num_pages}...")

                    # 1. Текст через pdfplumber
                    text = page.extract_text() or ""
                    text = text.strip()

                    # 2. Таблицы
                    tables_text = ""
                    if ENABLE_TABLES:
                        tables = page.extract_tables()
                        if tables:
                            tables_text = self._format_tables(tables)
                            logger.debug(f"      ✅ Найдено таблиц: {len(tables)}")

                    # 3. OCR, если текста мало
                    ocr_text = ""
                    if self.ocr and len(text) < 300:
                        logger.debug(f"      🔍 Мало текста ({len(text)} симв.), запускаю OCR...")
                        ocr_text = self._ocr_page_image(page)

                    # 4. Docling текст для этой страницы
                    docling_text = ""
                    if docling_page_texts and page_num in docling_page_texts:
                        docling_text = docling_page_texts[page_num]
                        logger.debug(f"      📑 Docling: добавлено {len(docling_text)} символов")

                    # 5. Объединяем всё
                    combined_text = "\n\n".join(
                        part for part in [text, tables_text, ocr_text, docling_text] if part and part.strip()
                    ).strip()

                    if not combined_text:
                        logger.debug(f"      ⚠️ Страница {page_num} пустая, пропускаю")
                        continue

                    # 6. Чистка через LLM
                    cleaned_text = self.text_cleaner.clean_text(
                        combined_text,
                        file_name=file_path.name,
                        page=page_num,
                    )

                    # 7. Чанки
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

    def _extract_with_docling(self, file_path: Path) -> Optional[Dict[int, str]]:
        """
        Извлечение текста с помощью Docling по страницам.
        Возвращает dict: {page_num: text}
        """
        if not self.use_docling or not self.docling_converter:
            return None

        try:
            logger.info(f"📑 Docling: конвертация файла {file_path.name}")
            result = self.docling_converter.convert(str(file_path))

            page_texts: Dict[int, str] = {}

            for page in result.document.pages:
                page_num = page.page_number or 0
                if page_num == 0:
                    continue

                if page_num > MAX_DOCLING_PAGES:
                    continue

                lines = []
                for block in page.blocks:
                    txt = block.to_text().strip()
                    if txt:
                        lines.append(txt)

                if lines:
                    page_texts[page_num] = "\n".join(lines)

            logger.info(f"📑 Docling: получено страниц с текстом: {len(page_texts)}")
            return page_texts if page_texts else None

        except Exception as e:
            logger.error(f"❌ Ошибка Docling при обработке {file_path.name}: {repr(e)}")
            return None

    def _ocr_page_image(self, page) -> str:
        """OCR страницы через PaddleOCR (из pdfplumber page)"""
        if not self.ocr:
            logger.debug("    ⚠️ OCR выключен (self.ocr is None)")
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
#km