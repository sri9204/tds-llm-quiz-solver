FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    git \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libgbm1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN playwright install --with-deps

ENV PORT=10000
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000", "--workers", "2", "--threads", "4"]
