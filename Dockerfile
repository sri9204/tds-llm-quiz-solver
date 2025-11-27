FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

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
    libgdk-pixbuf-2.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN playwright install chromium


CMD ["python", "app.py"]

