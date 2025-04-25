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

# Run the app
CMD ["python3", "app.py"]
