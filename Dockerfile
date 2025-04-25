FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up environment
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 3. Install Python packages first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app code
COPY app.py .

# 5. Create LoRA directory (but don't download during build)
RUN mkdir -p /lora

# 6. Runtime download (optional - better to mount volume)
# Uncomment ONLY if you want automatic downloads
# ENV LORA_REPO="ProfessionalLime1144/Iman_LoRA"
# COPY download_lora.py .
# CMD ["python", "download_lora.py"] && uvicorn app:app --host 0.0.0.0 --port 8000

# 7. Standard run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
