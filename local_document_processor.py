import logging
import re
import time
import psutil
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
    logger.warning("⚠️ PaddleOCR не установлен, OCR будет отключён.")

# Пытаемся подключить Docling
try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("⚠️ Docling не установлен, функционал Docling будет отключён.")


def log_system_stats(stage: str):
    """Логирование системных ресурсов"""
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024  # MB
    cpu_percent = process.cpu_percent(interval=0.1)

    logger.debug(f"📊 [{stage}] Память: {memory:.1f}MB, CPU: {cpu_percent:.1f}%")


def is_trash_text(text: str) -> bool:
    """
    Проверка, годится ли текст для индексации.
    Возвращает True, если текст — мусор (слишком мало, битая кодировка, артефакты).
    """
    if not text:
        return True

    cleaned = text.strip()
    if len(cleaned) < 50:  # Уменьшил до 50 символов
        return True

    # Доля нормальных букв vs остального
    letters = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", cleaned)
    ratio = len(letters) / max(len(cleaned), 1)
    if ratio < 0.2:  # Уменьшил до 20%
        # меньше 20% букв/цифр — похоже на мусор
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

        user_prompt = f"Файл: {file_name}, страница: {page}\n\nТекст:\n{text[:2000]}"  # Ограничиваем длину

        try:
            logger.debug(f"🧹 Отправка в LLM для чистки ({len(text)} символов)...")
            start_time = time.time()
            cleaned = self.ollama.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
            )
            clean_time = time.time() - start_time

            cleaned = cleaned.strip()
            logger.debug(f"✅ LLM очистка: {len(text)} → {len(cleaned)} символов за {clean_time:.1f}с")

            return cleaned
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
                logger.info("✅ Docling инициализирован (без ограничений по страницам)")
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
        start_time = time.time()

        try:
            logger.info(f"📄 НАЧИНАЮ ОБРАБОТКУ ФАЙЛА: {file_path.name}")

            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"📊 ФАЙЛ {file_path.name}: всего страниц: {num_pages}")
                log_system_stats(f"start_{file_path.name}")

                processed_pages = 0
                total_chunks = 0
                successful_pages = 0

                for page_num, page in enumerate(pdf.pages, start=1):
                    processed_pages += 1
                    logger.info(f"   📄 ОБРАБОТКА СТРАНИЦЫ {page_num}/{num_pages} файла {file_path.name}")

                    try:
                        # 1. БАЗОВЫЙ ТЕКСТ (pdfplumber)
                        text = page.extract_text() or ""
                        text = text.strip()
                        logger.info(f"      📝 pdfplumber извлек {len(text)} символов")

                        # 2. ТАБЛИЦЫ
                        tables_text = ""
                        if ENABLE_TABLES:
                            try:
                                tables = page.extract_tables()
                                if tables:
                                    tables_text = self._format_tables(tables)
                                    logger.info(f"      📊 Найдено таблиц: {len(tables)} ({len(tables_text)} символов)")
                            except Exception as e:
                                logger.warning(f"      ⚠️ Ошибка извлечения таблиц: {repr(e)}")

                        # Объединяем базовый текст + таблицы
                        combined_text = "\n\n".join(
                            part for part in [text, tables_text] if part and part.strip()).strip()
                        logger.info(f"      🔗 Объединенный текст: {len(combined_text)} символов")

                        # 3. ПРОВЕРКА: если текста мало или мусор → пробуем docling
                        if is_trash_text(combined_text):
                            logger.warning(
                                f"      ⚠️ МАЛО ТЕКСТА на стр. {page_num} файла {file_path.name} ({len(combined_text)} симв.), ПРОБУЕМ DOCLING...")

                            docling_text = self._extract_page_with_docling(file_path, page_num)
                            if docling_text and not is_trash_text(docling_text):
                                logger.info(f"      ✅ DOCLING УСПЕШЕН: {len(docling_text)} символов")
                                combined_text = docling_text
                            else:
                                logger.warning(f"      ⚠️ DOCLING НЕ ПОМОГ, ПРОБУЕМ OCR...")

                                # 4. OCR ПО КАРТИНКЕ СТРАНИЦЫ
                                if self.ocr:
                                    try:
                                        ocr_text = self._ocr_page_image(page, file_path.name, page_num)
                                        if ocr_text and not is_trash_text(ocr_text):
                                            logger.info(f"      ✅ OCR УСПЕШЕН: {len(ocr_text)} символов")
                                            combined_text = ocr_text
                                        else:
                                            logger.warning(f"      ❌ OCR ТОЖЕ НЕ ДАЛ НОРМАЛЬНОГО ТЕКСТА")
                                    except Exception as e:
                                        logger.error(f"      ❌ ОШИБКА OCR: {repr(e)}")
                                else:
                                    logger.warning(f"      ⚠️ OCR отключен")

                        # Если после всех попыток текста нет — пропускаем страницу
                        if not combined_text or is_trash_text(combined_text):
                            logger.warning(
                                f"      ⚠️ СТРАНИЦА {page_num} ФАЙЛА {file_path.name} ПУСТАЯ ПОСЛЕ ВСЕХ ПОПЫТОК, ПРОПУСКАЮ")
                            continue

                        # 5. ЧИСТКА ЧЕРЕЗ LLM
                        if ENABLE_TEXT_CLEANING:
                            logger.info(f"      🧹 Отправка в LLM для чистки...")
                            cleaned_text = self.text_cleaner.clean_text(
                                combined_text,
                                file_name=file_path.name,
                                page=page_num,
                            )
                            logger.info(f"      ✅ LLM ОЧИСТКА: {len(combined_text)} → {len(cleaned_text)} символов")
                        else:
                            cleaned_text = combined_text
                            logger.info(f"      ⚠️ ЧИСТКА LLM ОТКЛЮЧЕНА, использую исходный текст")

                        # 6. ЧАНКИ
                        logger.info(f"      ✂️ Разделение на чанки...")
                        chunks = self._split_into_chunks(cleaned_text)
                        logger.info(f"      ✅ РАЗБИТО НА {len(chunks)} ЧАНКОВ")
                        total_chunks += len(chunks)

                        for chunk in chunks:
                            fragments.append({
                                "content": chunk,
                                "page": page_num,
                                "type": "text",
                                "file": file_path.name,
                            })

                        successful_pages += 1
                        logger.info(f"      ✅ СТРАНИЦА {page_num} УСПЕШНО ОБРАБОТАНА")

                    except Exception as e:
                        logger.error(f"      ❌ КРИТИЧЕСКАЯ ОШИБКА НА СТРАНИЦЕ {page_num}: {repr(e)}")
                        continue

                    # Логируем каждую страницу
                    if page_num % 1 == 0:  # Логируем каждую страницу
                        log_system_stats(f"{file_path.name}_page_{page_num}")

                elapsed = time.time() - start_time
                logger.info(f"🎉 ФАЙЛ {file_path.name} ОБРАБОТАН ЗА {elapsed:.1f}с")
                logger.info(f"📊 ИТОГОВАЯ СТАТИСТИКА ДЛЯ {file_path.name}:")
                logger.info(f"   📄 Обработано страниц: {successful_pages}/{num_pages}")
                logger.info(f"   📦 Всего чанков: {total_chunks}")
                logger.info(f"   ⏱️ Общее время: {elapsed:.1f}с")
                if successful_pages > 0:
                    logger.info(f"   📈 Среднее время на страницу: {elapsed / successful_pages:.1f}с")
                    logger.info(f"   📈 Скорость: {successful_pages / elapsed:.1f} стр/сек")
                logger.info(f"   💾 Фрагментов для индексации: {len(fragments)}")

                return fragments

        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ОБРАБОТКЕ PDF {file_path.name}: {repr(e)}")
            return []

    def _extract_page_with_docling(self, file_path: Path, page_num: int) -> str:
        """
        Извлечение текста одной страницы через Docling.
        Возвращает текст страницы или пустую строку.
        """
        if not self.use_docling or not self.docling_converter:
            return ""

        try:
            logger.info(f"      🧠 Запуск Docling для страницы {page_num}...")
            result = self.docling_converter.convert(str(file_path))

            for page in result.document.pages:
                if page.page_number == page_num:
                    lines = []
                    for block in page.blocks:
                        txt = block.to_text().strip()
                        if txt:
                            lines.append(txt)

                    if lines:
                        docling_text = "\n".join(lines)
                        logger.info(f"      ✅ Docling извлек {len(docling_text)} символов")
                        return docling_text

            logger.info(f"      ⚠️ Docling не нашел текст на странице {page_num}")
            return ""

        except Exception as e:
            logger.error(f"      ❌ Ошибка Docling при обработке страницы {page_num} файла {file_path.name}: {repr(e)}")
            return ""

    def _ocr_page_image(self, page, file_name: str, page_num: int) -> str:
        """OCR страницы через PaddleOCR (из pdfplumber page)"""
        if not self.ocr:
            return ""

        try:
            # Конвертируем страницу в изображение
            logger.info(f"      🖼 Преобразование страницы {page_num} в изображение...")
            img = page.to_image(resolution=200).original  # Увеличил разрешение

            # Конвертируем PIL Image в numpy array
            import numpy as np
            img_np = np.array(img)

            # Если изображение имеет альфа-канал, конвертируем в RGB
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]

            logger.info(f"      🔠 Запуск PaddleOCR для страницы {page_num}...")
            result = self.ocr.ocr(img_np, cls=True)

            if not result or not result[0]:
                logger.info(f"      ⚠️ OCR не нашёл текста на странице {page_num}")
                return ""

            lines = []
            for line in result[0]:
                if len(line) > 1:
                    text = line[1][0]
                    if text and text.strip():
                        lines.append(text.strip())

            ocr_text = "\n".join(lines)
            logger.info(f"      ✅ OCR извлёк {len(ocr_text)} символов с страницы {page_num}")
            return ocr_text

        except Exception as e:
            logger.error(f"      ❌ Ошибка OCR на странице {page_num}: {repr(e)}")
            return ""

    def _format_tables(self, tables) -> str:
        """Форматирование таблиц в текст"""
        parts = []
        for table_idx, table in enumerate(tables, 1):
            try:
                for row_idx, row in enumerate(table):
                    row_text = []
                    for cell_idx, cell in enumerate(row):
                        if cell:
                            cell_text = str(cell).strip()
                            row_text.append(cell_text)
                    if row_text:
                        parts.append(" | ".join(row_text))
                parts.append("")  # Пустая строка между таблицами
            except Exception as e:
                logger.warning(f"      ⚠️ Ошибка форматирования таблицы {table_idx}: {repr(e)}")
        return "\n".join(parts)

    def _process_docx(self, file_path: Path) -> List[Dict]:
        """Обработка DOCX"""
        try:
            from docx import Document
        except ImportError:
            logger.error("❌ Для обработки DOCX нужен пакет python-docx (pip install python-docx)")
            return []

        fragments = []
        start_time = time.time()

        try:
            logger.info(f"📄 НАЧИНАЮ ОБРАБОТКУ DOCX: {file_path.name}")

            doc = Document(str(file_path))
            full_text = []

            for para in doc.paragraphs:
                txt = (para.text or "").strip()
                if txt:
                    full_text.append(txt)

            combined = "\n".join(full_text).strip()
            if not combined:
                logger.warning(f"⚠️ DOCX {file_path.name} пустой")
                return []

            logger.info(f"   📝 Исходный текст: {len(combined)} символов")

            if ENABLE_TEXT_CLEANING:
                cleaned_text = self.text_cleaner.clean_text(
                    combined,
                    file_name=file_path.name,
                    page=1,
                )
                logger.info(f"   ✅ LLM очистка: {len(combined)} → {len(cleaned_text)} символов")
            else:
                cleaned_text = combined
                logger.info(f"   ⚠️ ЧИСТКА LLM ОТКЛЮЧЕНА")

            logger.info(f"   ✂️ Разделение на чанки...")
            chunks = self._split_into_chunks(cleaned_text)

            for chunk in chunks:
                fragments.append({
                    "content": chunk,
                    "page": 1,
                    "type": "text",
                    "file": file_path.name,
                })

            elapsed = time.time() - start_time
            logger.info(f"✅ DOCX {file_path.name} обработан за {elapsed:.1f}с")
            logger.info(f"📊 Статистика: {len(chunks)} чанков")
            logger.info(f"📈 Скорость: {len(chunks) / elapsed:.1f} чанк/сек")

            return fragments

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке DOCX {file_path.name}: {repr(e)}")
            return []

    def _split_into_chunks(self, text: str) -> List[str]:
        """Режем текст на чанки с детальным логированием"""
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)

        logger.info(f"      ✂️ Начинаю разделение текста на чанки:")
        logger.info(f"         Длина текста: {length} символов")
        logger.info(f"         CHUNK_SIZE: {CHUNK_SIZE}")
        logger.info(f"         CHUNK_OVERLAP: {CHUNK_OVERLAP}")

        chunk_num = 1
        while start < length:
            end = min(start + CHUNK_SIZE, length)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)
                logger.info(f"         📦 Чанк {chunk_num}: позиции {start}-{end} ({len(chunk)} символов)")
                chunk_num += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

        logger.info(f"      ✅ Создано {len(chunks)} чанков")

        # Логируем примеры чанков
        if chunks:
            for i, chunk in enumerate(chunks[:3]):  # Показываем первые 3 чанка
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                logger.info(f"         📄 Чанк {i + 1} превью: '{preview}'")

        return chunks