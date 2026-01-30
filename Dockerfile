FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download models during build (cached in image layer)
COPY download_models.py .
RUN python download_models.py

# Copy FastAPI handler
COPY handler_fastapi.py .

# Expose port for Load Balancer mode
EXPOSE 8000

# Start FastAPI server
CMD ["python", "-u", "handler_fastapi.py"]
