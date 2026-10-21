FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml README.md config.yaml ./
COPY riskline riskline
RUN pip install --no-deps . \
 && python -m riskline.train \
 && rm -rf data \
 && useradd --create-home --uid 10001 app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "riskline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
