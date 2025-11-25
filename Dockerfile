FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow first (largest package)
RUN pip install --no-cache-dir --default-timeout=200 tensorflow==2.19.0

# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pillow==10.1.0 \
    numpy==1.26.2 \
    python-multipart==0.0.6 \
    jinja2==3.1.2 \
    requests==2.31.0

# Copy application code
COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
