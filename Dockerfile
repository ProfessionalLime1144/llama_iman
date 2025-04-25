FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid issues with older versions
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Retrying with specific versions for dependencies..." && \
     pip install --no-cache-dir torch==2.0.0 vllm==0.0.1 runpod==0.1.2 pydantic==1.10.2)

# Copy application code
COPY app.py .

# Copy fine-tuned LoRA weights
COPY fine_tuned_llama_lora /fine_tuned_llama_lora

# Run the app
CMD ["python3", "app.py"]
