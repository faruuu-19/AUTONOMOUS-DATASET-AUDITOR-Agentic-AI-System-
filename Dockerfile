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

# Hugging Face Spaces runs containers as UID 1000, so the app must own the
# directories it writes to at runtime (uploads, persisted job reports, and the
# meta-learning pickles). Render runs as root and is unaffected by this.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/backend/data/uploads /app/backend/reports/jobs \
    && chown -R appuser:appuser /app

USER appuser

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 1 --timeout 600 -b 0.0.0.0:${PORT:-7860} api_server:app"]
