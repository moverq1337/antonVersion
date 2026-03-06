# Mistral-only QR Service

Этот проект использует только Mistral AI (без локального OCR).

## Локальный запуск

1. Откройте терминал в папке проекта:
   `cd d:\разработки\mistral_only_service`
2. Создайте и активируйте venv:
   - `py -3 -m venv .venv`
   - `.venv\scripts\activate`
3. Установите зависимости:
   - `pip install -r requirements.txt`
4. Подготовьте `settings_ai` (если нет):
   - скопируйте `settings_ai.example` в `settings_ai`
   - вставьте ваш `MISTRAL_API_KEY`
5. Запустите сервис:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`
6. Откройте:
   - `http://localhost:8000`

## Деплой на Beget VPS (минимальными силами)

Подробно: см. `DEPLOY_BEGET_MIN.md`.
