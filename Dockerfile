FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create LoRA directory
RUN mkdir -p /lora

# Optional: Remove or customize the health check depending on RunPod setup
# HEALTHCHECK not strictly necessary for serverless RunPod
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Run with Python since runpod handles the serverless wrapping
CMD ["python3", "app.py"]
