FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy fine-tuned LoRA weights
COPY fine_tuned_llama_lora /fine_tuned_llama_lora

# Run the app
CMD ["python3", "app.py"]
