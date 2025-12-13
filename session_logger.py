"""
Логгер сессий пользователей.
Сохраняет стартовые данные и переписку в JSON-файлы.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

SESSIONS_FOLDER = Path("sessions")
SESSIONS_FOLDER.mkdir(exist_ok=True)


class SessionLogger:
    """Управление сессиями пользователей с сохранением на диск."""

    def __init__(self):
        self.sessions = {}  # user_id -> session_info

    def _get_user_id(self, user) -> str:
        """Получение ID пользователя."""
        return str(user.id)

    def start_session(self, user) -> str:
        """
        Создаёт новую сессию для пользователя, если её ещё нет.
        Возвращает session_id.
        """
        user_id = self._get_user_id(user)
        if user_id in self.sessions:
            return self.sessions[user_id]["session_id"]

        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_id = f"{start_time}_user_{user_id}"

        self.sessions[user_id] = {
            "session_id": session_id,
            "start_time": start_time,
            "user_id": user_id,
            "username": user.username or "unknown",
            "first_name": user.first_name or "",
            "initial_data": None,
            "messages": []
        }

        logger.info("🧾 Новая сессия: %s", session_id)
        self._flush_to_file(user_id)
        return session_id

    def set_initial_data(self, user, data_dict: Dict):
        """
        Сохраняем стартовые данные пользователя.
        data_dict: словарь с полями (contract, phone, model, etc.)
        """
        user_id = self._get_user_id(user)
        if user_id not in self.sessions:
            self.start_session(user)

        self.sessions[user_id]["initial_data"] = {
            "data": data_dict,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._flush_to_file(user_id)
        logger.info("📋 Стартовые данные сохранены для user %s", user_id)

    def add_messages(self, user, messages: List[Dict]):
        """
        Добавляем сообщения в лог (упрощённая переписка).
        messages: [{"role": "user"/"assistant", "content": "..."}, ...]
        """
        user_id = self._get_user_id(user)
        if user_id not in self.sessions:
            self.start_session(user)

        for msg in messages:
            self.sessions[user_id]["messages"].append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
            })

        self._flush_to_file(user_id)

    def _flush_to_file(self, user_id: str):
        """Сливаем данные сессии в файл JSON."""
        session = self.sessions.get(user_id)
        if not session:
            return

        session_id = session["session_id"]
        file_path = SESSIONS_FOLDER / f"{session_id}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("❌ Ошибка записи сессии %s: %s", session_id, repr(e))

    def get_session(self, user) -> Dict:
        """Получить текущую сессию пользователя."""
        user_id = self._get_user_id(user)
        return self.sessions.get(user_id, {})