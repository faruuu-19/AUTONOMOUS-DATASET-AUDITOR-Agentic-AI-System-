FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt gunicorn

COPY backend/ /app/backend/
COPY --from=frontend-build /app/frontend/dist/public /app/frontend/dist/public

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} api_server:app"]
