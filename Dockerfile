FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create LoRA directory
RUN mkdir -p /lora

# Copy application code
COPY app.py .

# Download LoRA adapter (optional - remove if mounting volume)
ARG LORA_REPO="ProfessionalLime1144/Iman_LoRA"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${LORA_REPO}', local_dir='/lora')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
