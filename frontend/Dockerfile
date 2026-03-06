FROM nginx:1.27-alpine

# Копируем статику фронтенда (контекст — корень проекта)
COPY static /usr/share/nginx/html/static
COPY static/index.html /usr/share/nginx/html/index.html

# Копируем конфиг nginx
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
