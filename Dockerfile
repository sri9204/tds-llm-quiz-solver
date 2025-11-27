FROM python:3.10-slim

# Prevents interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Playwright + PDF + audio
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

# Copy files
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Chromium for Playwright
RUN python -m playwright install chromium



CMD ["python", "app.py"]
