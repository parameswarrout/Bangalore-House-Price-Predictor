FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY ml_project ./ml_project
COPY backend ./backend

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/backend
ENV MODEL_DIR=/app/backend/models
ENV REQUIRE_MODELS=true

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
