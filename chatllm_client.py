import logging
import socket
import urllib.parse
from typing import List, Dict, Any

import requests

from config import (
    CHATLLM_API_KEY,
    CHATLLM_API_URL,
    CHATLLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    ENABLE_CHATLLM,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatLLMClient:
    """Клиент для работы с ChatLLM API (Abacus.AI) с расширенной диагностикой"""

    def __init__(self):
        self.api_key = CHATLLM_API_KEY
        self.api_url = CHATLLM_API_URL
        self.model = CHATLLM_MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS

    def _build_messages(self, user_query: str, context_chunks: List[Dict]) -> List[Dict[str, str]]:
        """Формирует messages для ChatLLM с учётом контекста (RAG)"""
        system_prompt = (
            "Ты техподдержка по технической документации. "
            "Отвечай чётко, структурировано и по делу, с опорой на предоставленные фрагменты."
        )

        context_texts = []
        for chunk in context_chunks:
            file = chunk.get("file", "Unknown")
            page = chunk.get("page", "N/A")
            content = chunk.get("content", "")
            context_texts.append(f"[FILE: {file} | PAGE: {page}]\n{content}")

        context_block = "\n\n---\n\n".join(context_texts) if context_texts else "Нет контекста."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Контекст из документации:\n{context_block}\n\n"
                    f"Вопрос пользователя:\n{user_query}"
                ),
            },
        ]
        return messages

    def generate_response(self, user_query: str, context_chunks: List[Dict]) -> str:
        """
        Основной метод: генерирует ответ на вопрос пользователя на основе контекста (RAG + ChatLLM).
        Если ChatLLM отключён или недоступен, возвращает понятное сообщение.
        """
        if not ENABLE_CHATLLM:
            logger.warning("ChatLLM отключён (ENABLE_CHATLLM=False)")
            return (
                "⚠️ AI-режим сейчас отключён на уровне конфигурации.\n\n"
                "Доступен только локальный поиск (`/search` или кнопка '🔍 Поиск')."
            )

        if not self.api_key:
            logger.error("CHATLLM_API_KEY не задан")
            return (
                "❌ AI-режим недоступен: не задан API-ключ ChatLLM.\n\n"
                "Проверь `.env` (переменная `CHATLLM_API_KEY`)."
            )

        messages = self._build_messages(user_query, context_chunks)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info("📡 Отправка запроса к ChatLLM: model=%s, url=%s", self.model, self.api_url)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            logger.error("❌ Ошибка сети при запросе к ChatLLM: %s", repr(e))
            return (
                "❌ Не удалось подключиться к ChatLLM API.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n\n"
                "Проверьте:\n"
                "• есть ли интернет у сервера,\n"
                "• не блокирует ли VPN/фаервол домен `apis.abacus.ai`,\n"
                "• корректен ли URL API (`CHATLLM_API_URL`)."
            )

        if response.status_code != 200:
            logger.error(
                "❌ Ошибка от ChatLLM API: HTTP %s, ответ: %s",
                response.status_code,
                response.text[:500],
            )
            return (
                "❌ ChatLLM API вернул ошибку.\n\n"
                f"HTTP статус: *{response.status_code}*\n"
                f"Тело ответа (обрезано): `{response.text[:400]}`\n\n"
                "Проверьте:\n"
                "• правильность `CHATLLM_API_KEY`,\n"
                "• имя модели (`CHATLLM_MODEL`),\n"
                "• URL (`CHATLLM_API_URL`)."
            )

        try:
            data = response.json()
        except ValueError as e:
            logger.error("❌ Не удалось распарсить JSON от ChatLLM: %s", repr(e))
            return (
                "❌ Не удалось прочитать ответ от ChatLLM API (невалидный JSON).\n\n"
                f"Техническая ошибка: `{repr(e)}`\n"
                f"Сырой ответ (обрезано): `{response.text[:400]}`"
            )

        try:
            content = data["choices"][0]["message"]["content"]
            logger.info("✅ Ответ от ChatLLM успешно получен")
            return content
        except Exception as e:
            logger.error("❌ Неожиданная структура ответа ChatLLM: %s", repr(e))
            return (
                "❌ Неожиданный формат ответа от ChatLLM API.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n"
                f"JSON (обрезано): `{str(data)[:400]}`"
            )

    def generate_response_with_history(self, history: List[Dict], user_query: str, context_docs: List[Dict]) -> str:
        """
        Генерация ответа с учётом истории диалога.

        Args:
            history: История диалога [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            user_query: Текущий запрос пользователя
            context_docs: Релевантные фрагменты из RAG

        Returns:
            Ответ модели
        """
        # Формируем контекст из документов
        context_text = "\n\n".join([
            f"[Источник: {doc.get('file', 'Unknown')}, стр. {doc.get('page', 'N/A')}]\n{doc.get('text', '')}"
            for doc in context_docs[:5]
        ])

        # Системный промпт
        system_prompt = f"""Ты — AI-ассистент для работы с технической документацией по лифтам и лифтовому оборудованию.

    Твоя задача:
    - Отвечать на вопросы пользователя, используя предоставленный контекст из документов.
    - Учитывать историю диалога для понимания контекста беседы.
    - Если в истории есть ссылки на предыдущие темы ("как я говорил выше", "а что насчёт того"), используй их.
    - Если информации нет в контексте — честно скажи об этом.
    - Отвечай кратко, по делу, на русском языке.
    - Используй технический язык, но понятный для инженеров и монтажников.

    Контекст из документов:
    {context_text}
    """

        # Собираем messages: system + история + текущий запрос
        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю
        messages.extend(history)

        # Добавляем текущий запрос
        messages.append({"role": "user", "content": user_query})

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.model:
            payload["model"] = self.model

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                error_msg = data.get("error", "Unknown error")
                logger.error("❌ ChatLLM API вернул ошибку: %s", error_msg)
                return f"❌ Ошибка API: {error_msg}"

            answer = data.get("content", "").strip()
            if not answer:
                logger.warning("⚠️ ChatLLM вернул пустой ответ")
                return "⚠️ Модель вернула пустой ответ"

            return answer

        except requests.exceptions.Timeout:
            logger.error("❌ Таймаут при запросе к ChatLLM API")
            return "❌ Превышено время ожидания ответа от API"
        except requests.exceptions.RequestException as e:
            logger.error("❌ Ошибка при запросе к ChatLLM API: %s", repr(e))
            return f"❌ Ошибка соединения с API: {str(e)}"
        except Exception as e:
            logger.error("❌ Неожиданная ошибка: %s", repr(e))
            return f"❌ Внутренняя ошибка: {str(e)}"

    def generate_clarification_questions(self, user_query: str, context_chunks: List[Dict]) -> List[str]:
        """
        Генерирует уточняющие вопросы, если запрос пользователя неоднозначный.
        Возвращает список коротких вопросов без пояснений.
        Количество вопросов определяет сама модель.
        """
        if not ENABLE_CHATLLM or not self.api_key:
            return []

        system_prompt = (
            "Ты помощник технической поддержки. "
            "Пользователь задал неоднозначный вопрос. "
            "Сгенерируй несколько коротких уточняющих вопросов (каждый на отдельной строке), "
            "чтобы понять, что именно его интересует. "
            "Вопросы должны быть ОЧЕНЬ короткими (максимум 5-7 слов), без пояснений и деталей после двоеточия."
        )

        context_texts = []
        for chunk in context_chunks[:3]:  # берём только первые 3 фрагмента для экономии токенов
            file = chunk.get("file", "Unknown")
            content = chunk.get("content", "")[:200]  # обрезаем контент
            context_texts.append(f"[{file}]: {content}")

        context_block = "\n".join(context_texts) if context_texts else "Нет контекста."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context_block}\n\n"
                    f"Вопрос пользователя: {user_query}\n\n"
                    f"Сгенерируй несколько коротких уточняющих вопросов (каждый на новой строке)."
                ),
            },
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 200,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info("📡 Генерация уточняющих вопросов для: %s", user_query)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60,
            )

            if response.status_code != 200:
                logger.error("❌ Ошибка при генерации вопросов: HTTP %s", response.status_code)
                return []

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Парсим вопросы (каждый на новой строке)
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            # Убираем нумерацию, если модель её добавила
            questions = [q.lstrip('0123456789.-) ') for q in questions]

            logger.info("✅ Сгенерировано %d уточняющих вопросов", len(questions))
            return questions  # возвращаем все вопросы, которые сгенерировала модель

        except Exception as e:
            logger.error("❌ Ошибка при генерации уточняющих вопросов: %s", repr(e))
            return []

    # ==== Диагностика ====

    def test_connection(self) -> Dict[str, str]:
        """
        Полный тест подключения к ChatLLM без проверки 8.8.8.8.
        1) Проверка DNS для apis.abacus.ai
        2) Тестовый запрос к ChatLLM API (если есть ключ)
        """
        if not ENABLE_CHATLLM:
            msg = (
                "⚠️ AI-режим сейчас отключён (ENABLE_CHATLLM=False).\n\n"
                "Включите его в `.env`, установив `ENABLE_CHATLLM=true`."
            )
            logger.warning(msg)
            return {"ok": "false", "message": msg}

        if not self.api_key:
            msg = (
                "❌ CHATLLM_API_KEY не задан.\n\n"
                "Укажите API-ключ в `.env` и перезапустите бота."
            )
            logger.error(msg)
            return {"ok": "false", "message": msg}

        # 1. Проверка DNS для apis.abacus.ai
        dns_ok, dns_msg = self._check_dns_for_apis()
        if not dns_ok:
            return {"ok": "false", "message": dns_msg}

        # 2. Тестовый запрос к ChatLLM API
        api_ok, api_msg = self._check_chatllm_api()
        return {"ok": "true" if api_ok else "false", "message": api_msg}

    def _check_dns_for_apis(self) -> (bool, str):
        """Проверяем, резолвится ли хост из CHATLLM_API_URL"""
        parsed = urllib.parse.urlparse(self.api_url)
        host = parsed.hostname or "routellm.abacus.ai"
        logger.info("🔍 Проверка DNS для %s ...", host)

        try:
            ip = socket.gethostbyname(host)
            msg = f"✅ DNS OK: {host} -> {ip}"
            logger.info(msg)
            return True, msg
        except socket.gaierror as e:
            msg = (
                f"❌ DNS не может разрешить `{host}`.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n\n"
                "Это проблема сети/провайдера/VPN, а не кода бота.\n"
                "Нужно:\n"
                "• настроить нормальный DNS,\n"
                "• либо запустить бота в сети, где этот хост доступен."
            )
            logger.error(msg)
            return False, msg

    def _check_chatllm_api(self) -> (bool, str):
        """Тестовый запрос к ChatLLM API"""
        logger.info("📡 Тестовый запрос к ChatLLM API: %s", self.api_url)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
            "temperature": 0.0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            msg = (
                "❌ Не удалось подключиться к ChatLLM API.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n\n"
                "Проверьте:\n"
                "• доступность интернета/выхода наружу,\n"
                "• не блокирует ли VPN/фаервол `apis.abacus.ai`,\n"
                "• корректность `CHATLLM_API_URL`."
            )
            logger.error(msg)
            return False, msg

        if resp.status_code != 200:
            msg = (
                "❌ ChatLLM API ответил с ошибкой.\n\n"
                f"HTTP статус: *{resp.status_code}*\n"
                f"Тело ответа (обрезано): `{resp.text[:400]}`\n\n"
                "Проверьте:\n"
                "• правильность `CHATLLM_API_KEY`,\n"
                "• имя модели (`CHATLLM_MODEL`),\n"
                "• URL (`CHATLLM_API_URL`)."
            )
            logger.error(msg)
            return False, msg

        try:
            data = resp.json()
            _ = data["choices"][0]["message"]["content"]
        except Exception as e:
            msg = (
                "❌ Неожиданный формат ответа от ChatLLM API.\n\n"
                f"Техническая ошибка: `{repr(e)}`\n"
                f"JSON (обрезано): `{str(resp.text)[:400]}`"
            )
            logger.error(msg)
            return False, msg

        msg = "✅ Подключение к ChatLLM API успешно, тестовый запрос прошёл."
        logger.info(msg)
        return True, msg

    def generate_response_with_history(self, history: List[Dict], user_query: str, context_docs: List[Dict]) -> str:
        """
        Генерация ответа с учётом истории диалога.
        Использует существующий метод generate_response, чтобы не дублировать логику работы с API.

        Args:
            history: История диалога [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            user_query: Текущий запрос пользователя
            context_docs: Релевантные фрагменты из RAG

        Returns:
            Ответ модели
        """
        # Собираем историю в текстовый блок
        history_lines = []

        for msg in history[-20:]:  # берём последние 20 сообщений (перестраховка)
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "").strip()
            if not content:
                continue

            if role == "user":
                prefix = "Пользователь:"
            elif role == "assistant":
                prefix = "Ассистент:"
            else:
                prefix = "Сообщение:"

            history_lines.append(f"{prefix} {content}")

        history_text = "\n".join(history_lines).strip()

        # Формируем расширенный запрос с учётом истории
        if history_text:
            extended_query = (
                "Контекст беседы до текущего вопроса:\n"
                f"{history_text}\n\n"
                "Текущий вопрос пользователя:\n"
                f"{user_query}"
            )
        else:
            extended_query = user_query

        # Используем существующий метод generate_response
        return self.generate_response(extended_query, context_docs)