FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and imutils
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p uploads results models/classifier models/segmentation

# Expose port (Render provides $PORT)
ENV PORT=8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
