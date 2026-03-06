from __future__ import annotations

import base64
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from app.config import settings


@dataclass
class OCRResult:
    text: str
    confidence: float


class OCRProvider(ABC):
    @abstractmethod
    def read_wb(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError

    @abstractmethod
    def read_code(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError


class MistralAIProvider(OCRProvider):
    def __init__(self, api_key: str, model: str, prompt_wb: str, prompt_code: str):
        self.api_key = api_key
        self.model = model
        self.prompt_wb = prompt_wb
        self.prompt_code = prompt_code
        self._last_call = {
            'provider': 'mistral_ai',
            'model': model,
            'request_id': None,
            'status_code': None,
            'error': None,
        }

    def read_wb(self, image: np.ndarray) -> OCRResult:
        text, conf = self._extract_text(image, prompt=self.prompt_wb)
        cleaned = re.sub(r'[^A-Z]', '', text.upper())
        return OCRResult(text=cleaned, confidence=conf)

    def read_code(self, image: np.ndarray) -> OCRResult:
        text, conf = self._extract_text(image, prompt=self.prompt_code)
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return OCRResult(text=cleaned, confidence=conf)

    def get_last_call(self) -> dict:
        return dict(self._last_call)

    def _extract_text(self, image: np.ndarray, prompt: str) -> tuple[str, float]:
        ok, buffer = cv2.imencode('.png', image)
        if not ok:
            self._last_call.update({'request_id': None, 'status_code': None, 'error': 'encode_failed'})
            return '', 0.0

        image_b64 = base64.b64encode(buffer.tobytes()).decode('ascii')
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': f'data:image/png;base64,{image_b64}'},
                    ],
                }
            ],
            'temperature': 0,
        }

        try:
            response = requests.post('https://api.mistral.ai/v1/chat/completions', headers=headers, json=payload, timeout=20)
            status_code = response.status_code
            request_id = response.headers.get('x-request-id') or response.headers.get('x-mistral-request-id')
            response.raise_for_status()

            body = response.json()
            text = body['choices'][0]['message']['content']
            if isinstance(text, list):
                text = ''.join(part.get('text', '') for part in text if isinstance(part, dict))

            self._last_call.update({'request_id': request_id, 'status_code': status_code, 'error': None})
            return str(text).strip(), 0.9
        except Exception as ex:
            status_code = None
            request_id = None
            if isinstance(ex, requests.HTTPError) and ex.response is not None:
                status_code = ex.response.status_code
                request_id = ex.response.headers.get('x-request-id') or ex.response.headers.get('x-mistral-request-id')
            self._last_call.update({'request_id': request_id, 'status_code': status_code, 'error': type(ex).__name__})
            return '', 0.0


def _read_settings_from_file() -> dict[str, str]:
    out: dict[str, str] = {}
    key_file = Path.cwd() / 'settings_ai'
    if not key_file.exists():
        return out

    try:
        for raw in key_file.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                out[k.strip().upper()] = v.strip()
            elif 'MISTRAL_API_KEY' not in out:
                out['MISTRAL_API_KEY'] = line
    except Exception:
        return out

    return out


def _resolve_api_key() -> str:
    api_key = (settings.mistral_api_key or '').strip()
    if api_key:
        return api_key
    cfg = _read_settings_from_file()
    return cfg.get('MISTRAL_API_KEY', '').strip()


def _resolve_ai_model() -> str:
    cfg = _read_settings_from_file()
    from_file = cfg.get('MISTRAL_AI_MODEL', '').strip()
    if from_file:
        return from_file
    return settings.mistral_model


def _resolve_prompt_wb() -> str:
    cfg = _read_settings_from_file()
    value = cfg.get('MISTRAL_PROMPT_WB', '').strip()
    if value:
        return value
    return 'Read exactly the two red letters above the largest QR code. Return only letters.'


def _resolve_prompt_code() -> str:
    cfg = _read_settings_from_file()
    value = cfg.get('MISTRAL_PROMPT_CODE', '').strip()
    if value:
        return value
    return 'Read the alphanumeric code below the same largest QR code. Return only A-Z and 0-9.'


def build_mistral_ai_provider() -> Optional[OCRProvider]:
    api_key = _resolve_api_key()
    if not api_key:
        return None
    return MistralAIProvider(
        api_key=api_key,
        model=_resolve_ai_model(),
        prompt_wb=_resolve_prompt_wb(),
        prompt_code=_resolve_prompt_code(),
    )


def build_fallback_provider() -> Optional[OCRProvider]:
    return None
