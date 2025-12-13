"""
Тестирование OCR: поиск конкретного текста в PDF
Использование:
    python test_ocr_search.py <имя_файла.pdf> <поисковый_запрос>

Пример:
    python test_ocr_search.py "Veda LCS(User Manual)DS0001 Rev300724.pdf" "СУК-1"
    python test_ocr_search.py "ZAA21310BZ_SUR_MKC220_GOST33984_30.10.2025.pdf" "адресация"
"""

import sys
from pathlib import Path
import pdfplumber
from paddleocr import PaddleOCR

# Инициализация OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ru', use_gpu=False, show_log=False)


def ocr_page_image(page):
    """OCR страницы через PaddleOCR"""
    try:
        img = page.to_image(resolution=150).original
        result = ocr.ocr(img, cls=True)

        if not result or not result[0]:
            return ""

        lines = []
        for line in result[0]:
            text = line[1][0] if len(line) > 1 else ""
            if text:
                lines.append(text)

        return "\n".join(lines)

    except Exception as e:
        print(f"    ❌ Ошибка OCR: {repr(e)}")
        return ""


def search_in_pdf(pdf_path: Path, search_query: str):
    """Поиск текста в PDF (pdfplumber + OCR)"""
    search_lower = search_query.lower()

    print(f"\n📄 Файл: {pdf_path.name}")
    print(f"🔍 Ищем: '{search_query}'\n")
    print("=" * 80)

    found_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        print(f"📚 Всего страниц: {num_pages}\n")

        for page_num, page in enumerate(pdf.pages, start=1):
            # 1. Текст через pdfplumber
            text = page.extract_text() or ""
            text = text.strip()

            # 2. OCR (если мало текста)
            ocr_text = ""
            if len(text) < 300:
                print(f"   [Стр. {page_num}] Мало текста ({len(text)} симв.), запускаю OCR...")
                ocr_text = ocr_page_image(page)

            # 3. Объединяем
            combined = f"{text}\n\n{ocr_text}".strip()

            # 4. Ищем
            if search_lower in combined.lower():
                found_count += 1
                print(f"\n✅ НАЙДЕНО на странице {page_num}:")
                print("-" * 80)

                # Показываем контекст (±200 символов вокруг найденного)
                idx = combined.lower().find(search_lower)
                start = max(0, idx - 200)
                end = min(len(combined), idx + len(search_query) + 200)

                snippet = combined[start:end]
                # Подсвечиваем найденное (заглавными)
                snippet_highlighted = snippet[:idx - start] + f">>>{search_query.upper()}<<<" + snippet[
                    idx - start + len(search_query):]

                print(snippet_highlighted)
                print("-" * 80)

    print(f"\n📊 Итого: найдено на {found_count} страницах из {num_pages}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Использование:")
        print("    python test_ocr_search.py <имя_файла.pdf> <поисковый_запрос>")
        print("\nПример:")
        print('    python test_ocr_search.py "Veda LCS(User Manual)DS0001 Rev300724.pdf" "СУК-1"')
        sys.exit(1)

    file_name = sys.argv[1]
    search_query = " ".join(sys.argv[2:])

    pdf_path = Path("documents") / file_name

    if not pdf_path.exists():
        print(f"❌ Файл не найден: {pdf_path}")
        sys.exit(1)

    search_in_pdf(pdf_path, search_query)