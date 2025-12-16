"""
Клиент для работы с Ollama API (локальная LLM)
"""

import requests
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Клиент для работы с Ollama"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

        logger.info(f"🤖 Ollama клиент: {self.base_url}, модель: {self.model}")

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
        """Генерация текста через Ollama (/api/generate)"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            logger.debug(f"🤖 Отправка запроса к Ollama: URL={url}")
            logger.debug(f"Payload (укорочено): {str(payload)[:500]}")

            response = requests.post(
                url,
                json=payload,
                timeout=180,  # 3 минуты
            )
            response.raise_for_status()

            result = response.json()
            answer = (result.get("response") or "").strip()

            logger.debug(f"🤖 Ответ от Ollama: {len(answer)} символов")
            return answer

        except requests.exceptions.Timeout as e:
            logger.error(f"❌ Таймаут запроса к Ollama: {repr(e)}")
            raise Exception(f"Таймаут связи с Ollama: {e}")
        except requests.exceptions.HTTPError as e:
            status = response.status_code if "response" in locals() else "no_response"
            text = response.text[:500] if "response" in locals() else ""
            logger.error(f"❌ HTTP ошибка Ollama: {repr(e)}, status={status}, body={text}")
            raise Exception(f"Ошибка связи с Ollama: {e}")
        except Exception as e:
            logger.error(f"❌ Общая ошибка запроса к Ollama: {repr(e)}")
            raise Exception(f"Ошибка связи с Ollama: {e}")

    def test_connection(self) -> bool:
        """Проверка доступности Ollama и наличия нужной модели"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.model not in model_names:
                logger.warning(
                    f"⚠️ Модель {self.model} не найдена в Ollama. "
                    f"Доступные модели: {model_names}"
                )
                return False

            logger.info(f"✅ Ollama доступен, модель {self.model} найдена")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Ollama: {repr(e)}")
            return False