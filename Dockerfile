FROM python:3.10-slim

WORKDIR /app

# --- system libs needed by faiss-cpu (keep) ---
RUN apt-get update && apt-get install -y \
        build-essential cmake libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# --- install Python deps first (caches better) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- copy code & data ---
COPY app/ ./app
COPY data/ ./data

# ensure /app/app is a Python package
RUN touch app/__init__.py

ENV PORT=8080
EXPOSE 8080

# <-- module path is app.main, not main
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
