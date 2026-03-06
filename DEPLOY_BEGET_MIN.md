# DEPLOY на Beget VPS (очень просто)

Ниже вариант для Ubuntu 22.04/24.04 на VPS.

## 0) Что нужно заранее

- VPS от Beget
- Домен или субдомен (например `scan.yourdomain.ru`)
- Доступ SSH (логин `root` + пароль)

## 1) Подключиться к серверу

На Windows откройте PowerShell:

```powershell
ssh root@IP_ВАШЕГО_VPS
```

## 2) Установить пакеты

```bash
apt update
apt install -y python3 python3-venv python3-pip nginx
```

## 3) Залить проект на сервер

### Вариант A (самый простой): через WinSCP

1. Подключитесь к серверу по SFTP (`root`, IP, пароль).
2. Скопируйте папку `mistral_only_service` в `/opt/mistral_only_service`.

### Вариант B: через scp

На вашем ПК (PowerShell):

```powershell
scp -r D:\Разработки\mistral_only_service root@IP_ВАШЕГО_VPS:/opt/
```

## 4) Подготовить Python и зависимости

На сервере:

```bash
cd /opt/mistral_only_service
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Проверьте файл `/opt/mistral_only_service/settings_ai`:
- `MISTRAL_API_KEY=...`
- `MISTRAL_AI_MODEL=...`
- промпты при необходимости.

## 5) Создать systemd-сервис

```bash
cat > /etc/systemd/system/mistral-qr.service << 'EOF'
[Unit]
Description=Mistral QR FastAPI Service
After=network.target

[Service]
User=root
WorkingDirectory=/opt/mistral_only_service
ExecStart=/opt/mistral_only_service/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mistral-qr
systemctl start mistral-qr
systemctl status mistral-qr --no-pager
```

## 6) Настроить Nginx

```bash
cat > /etc/nginx/sites-available/mistral-qr << 'EOF'
server {
    listen 80;
    server_name scan.yourdomain.ru;

    client_max_body_size 15M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/mistral-qr /etc/nginx/sites-enabled/mistral-qr
nginx -t
systemctl reload nginx
```

## 7) Привязать домен/субдомен

У регистратора/в DNS:
- создайте `A` запись:
  - `scan` -> `IP_ВАШЕГО_VPS`

Подождите 5-30 минут (иногда до 24 часов).

## 8) Включить HTTPS (очень желательно)

```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d scan.yourdomain.ru
```

После успеха клиентская ссылка:
- `https://scan.yourdomain.ru`

## 9) Как дать ссылку клиенту

Просто отправляете URL:
- `https://scan.yourdomain.ru`

Клиент открывает в телефоне/браузере и тестирует.

## 10) Если у вас сайт на Tilda

Самый простой путь:
- на Tilda сделать кнопку "Сканер" со ссылкой на `https://scan.yourdomain.ru`.

Почему так лучше:
- камера в iframe работает хуже и часто блокируется браузером.
- отдельная страница надежнее для мобильной камеры.

## Полезные команды

Перезапуск сервиса:
```bash
systemctl restart mistral-qr
```

Логи сервиса:
```bash
journalctl -u mistral-qr -n 200 --no-pager
```

Проверка Nginx:
```bash
nginx -t
```
