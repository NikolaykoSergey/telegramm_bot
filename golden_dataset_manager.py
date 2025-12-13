"""
Менеджер золотого датасета (Ground Truth)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GoldenDatasetManager:
    """Управление золотым датасетом вопросов-ответов"""

    def __init__(self, dataset_path: str = "golden_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dict:
        """Загрузка датасета"""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки golden_dataset.json: {repr(e)}")
                return self._create_empty_dataset()
        else:
            return self._create_empty_dataset()

    def _create_empty_dataset(self) -> Dict:
        """Создание пустого датасета"""
        return {
            "version": "1.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "questions": []
        }

    def _save_dataset(self):
        """Сохранение датасета"""
        try:
            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Golden dataset сохранён: {len(self.dataset['questions'])} вопросов")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения golden_dataset.json: {repr(e)}")

    def add_question(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        user_id: int,
        feedback: str = "helpful"
    ):
        """
        Добавление вопроса в датасет

        Args:
            question: Вопрос пользователя
            answer: Ответ бота
            sources: Список источников [{"file": "...", "page": ..., "score": ...}]
            user_id: ID пользователя Telegram
            feedback: "helpful" или "not_helpful"
        """
        # Проверяем, нет ли уже такого вопроса
        for q in self.dataset["questions"]:
            if q["question"].lower().strip() == question.lower().strip():
                logger.info(f"⚠️ Вопрос уже есть в датасете: {question[:50]}...")
                return

        # Формируем новую запись
        new_id = max([q["id"] for q in self.dataset["questions"]], default=0) + 1

        entry = {
            "id": new_id,
            "question": question,
            "expected_answer": answer,
            "source_file": sources[0]["file"] if sources else "unknown",
            "source_page": sources[0]["page"] if sources else 0,
            "sources": sources,
            "keywords": self._extract_keywords(question),
            "difficulty": "unknown",
            "category": "user_feedback",
            "feedback": feedback,
            "user_id": user_id,
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.dataset["questions"].append(entry)
        self._save_dataset()

        logger.info(f"✅ Добавлен вопрос #{new_id} в golden dataset")

    def _extract_keywords(self, text: str) -> List[str]:
        """Простое извлечение ключевых слов (можно улучшить)"""
        # Убираем стоп-слова и берём уникальные слова длиннее 3 символов
        stop_words = {"как", "что", "где", "когда", "почему", "какой", "какая", "какие", "для", "при", "это", "или"}
        words = text.lower().split()
        keywords = [w.strip(".,?!:;") for w in words if len(w) > 3 and w not in stop_words]
        return list(set(keywords))[:10]  # Максимум 10 ключевых слов

    def get_all_questions(self) -> List[Dict]:
        """Получить все вопросы"""
        return self.dataset["questions"]

    def get_stats(self) -> Dict:
        """Статистика датасета"""
        questions = self.dataset["questions"]
        return {
            "total": len(questions),
            "helpful": len([q for q in questions if q.get("feedback") == "helpful"]),
            "not_helpful": len([q for q in questions if q.get("feedback") == "not_helpful"]),
            "categories": list(set([q.get("category", "unknown") for q in questions]))
        }

#