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
    def __init__(self, api_key: str, model: str, prompt: str):
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self._last_call = {
            'provider': 'mistral_ai',
            'model': model,
            'request_id': None,
            'status_code': None,
            'error': None,
        }

    def read_wb(self, image: np.ndarray) -> OCRResult:
        # Single-prompt mode: WB detected locally for speed/stability.
        if image is None or image.size == 0:
            return OCRResult(text='', confidence=0.0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (140, 45, 45), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 45, 45), (12, 255, 255))
        red = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), dtype=np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel, iterations=1)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=1)

        red_ratio = float(np.count_nonzero(red)) / float(red.size)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
        components = 0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= 30:
                components += 1

        if red_ratio >= 0.008 and 1 <= components <= 8:
            conf = min(0.95, 0.55 + red_ratio * 8.0)
            return OCRResult(text='WB', confidence=conf)

        return OCRResult(text='', confidence=0.0)

    def read_code(self, image: np.ndarray) -> OCRResult:
        text, conf = self._extract_text(image, prompt=self.prompt)
        digits = ''.join(ch for ch in str(text) if ch.isdigit())
        # Keep realistic length window for this label format.
        if len(digits) > 16:
            digits = digits[:16]
        return OCRResult(text=digits, confidence=conf)

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


def _resolve_prompt() -> str:
    cfg = _read_settings_from_file()
    value = cfg.get('MISTRAL_PROMPT', '').strip()
    if value:
        return value
    return 'Write a numeric code, which is located in two lines immediately below the largest QR code, above which there are two red letters WB. Return digits only.'


def build_mistral_ai_provider() -> Optional[OCRProvider]:
    api_key = _resolve_api_key()
    if not api_key:
        return None
    return MistralAIProvider(
        api_key=api_key,
        model=_resolve_ai_model(),
        prompt=_resolve_prompt(),
    )


def build_fallback_provider() -> Optional[OCRProvider]:
    return None
