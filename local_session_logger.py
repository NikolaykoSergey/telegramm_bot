import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from local_config import SESSIONS_FOLDER

logger = logging.getLogger(__name__)


class SessionLogger:
    """Логирование сессий пользователей"""

    def __init__(self):
        self.sessions_folder = SESSIONS_FOLDER
        self.sessions_folder.mkdir(exist_ok=True)
        logger.info(f"✅ SessionLogger инициализирован (папка: {self.sessions_folder})")

    def start_session(self, user):
        """Начало новой сессии"""
        user_id = user.id
        username = user.username or "unknown"
        full_name = user.full_name or "Unknown User"

        session_file = self.sessions_folder / f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        session_data = {
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "start_time": datetime.now().isoformat(),
            "initial_data": {},
            "messages": [],
            "feedback": []
        }

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📝 Новая сессия: {session_file.name}")

        except Exception as e:
            logger.error(f"❌ Ошибка создания сессии: {repr(e)}")

    def set_initial_data(self, user, data: Dict):
        """Сохранение начальных данных"""
        user_id = user.id
        session_file = self._get_latest_session_file(user_id)

        if not session_file:
            return

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            session_data["initial_data"] = data

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📝 Начальные данные сохранены для пользователя {user_id}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения начальных данных: {repr(e)}")

    def add_messages(self, user, messages: List[Dict]):
        """Добавление сообщений в сессию"""
        user_id = user.id
        session_file = self._get_latest_session_file(user_id)

        if not session_file:
            return

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            session_data["messages"].extend(messages)

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ Ошибка добавления сообщений: {repr(e)}")

    def log_feedback(self, user, feedback_type: str, details: str):
        """Логирование обратной связи"""
        user_id = user.id
        session_file = self._get_latest_session_file(user_id)

        if not session_file:
            return

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            if 'feedback' not in session_data:
                session_data['feedback'] = []

            session_data['feedback'].append({
                'type': feedback_type,
                'details': details,
                'timestamp': datetime.now().isoformat()
            })

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📝 Обратная связь сохранена: {feedback_type}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения обратной связи: {repr(e)}")

    def _get_latest_session_file(self, user_id: int) -> Optional[Path]:
        """Получение последнего файла сессии пользователя"""
        session_files = list(self.sessions_folder.glob(f"session_{user_id}_*.json"))

        if not session_files:
            return None

        # Сортируем по времени создания
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return session_files[0]

#