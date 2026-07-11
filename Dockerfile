FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27"

COPY app.py .

ENV PORT=8100
EXPOSE 8100

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
