"""
Тестирование RAG-системы по золотому датасету
"""

import json
import logging
from pathlib import Path
from local_rag_system import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_golden_dataset():
    """Прогон всех вопросов из golden_dataset.json"""
    dataset_path = Path("golden_dataset.json")

    if not dataset_path.exists():
        print("❌ Файл golden_dataset.json не найден")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])

    if not questions:
        print("⚠️ Датасет пуст")
        return

    print(f"📊 Тестирование {len(questions)} вопросов из golden dataset\n")
    print("=" * 80)

    rag = RAGSystem()

    results = []

    for q in questions:
        question = q["question"]
        expected = q.get("expected_answer", "")

        print(f"\n❓ Вопрос #{q['id']}: {question}")
        print(f"✅ Ожидаемый ответ: {expected[:200]}...")

        try:
            result = rag.query(question)
            answer = result["answer"]
            relevance = result.get("relevance", 0)

            print(f"🤖 Ответ бота: {answer[:200]}...")
            print(f"📊 Релевантность: {relevance:.1f}%")

            # Простая оценка (можно улучшить через эмбеддинги или LLM)
            is_correct = "MANUAL_CHECK"  # Требует ручной проверки

            results.append({
                "id": q["id"],
                "question": question,
                "expected": expected,
                "actual": answer,
                "relevance": relevance,
                "status": is_correct
            })

        except Exception as e:
            print(f"❌ Ошибка: {repr(e)}")
            results.append({
                "id": q["id"],
                "question": question,
                "status": "ERROR",
                "error": str(e)
            })

        print("-" * 80)

    # Сохраняем результаты
    results_path = Path("test_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {results_path}")
    print(f"📊 Протестировано: {len(results)} вопросов")


if __name__ == "__main__":
    test_golden_dataset()