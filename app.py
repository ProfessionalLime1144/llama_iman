import logging
import os
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

# ===== Logging Setup =====
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s [%(levelname)s] %(message)s',
  handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===== Configuration =====
class Config:
  BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3-8B-Instruct")
  LORA_PATH = "/lora"  # RunPod volume mount path
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  MAX_TOKENS_LIMIT = 2048

# ===== API Setup =====
app = FastAPI(title="Llama LoRA API")

# ===== Model Initialization =====
try:
  logger.info(f"🔥 Initializing vLLM engine on {Config.DEVICE}")
  start_time = time.time()
  
  engine = LLM(
    model=Config.BASE_MODEL,
    enable_lora=True,
    max_lora_rank=64,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=Config.MAX_TOKENS_LIMIT
  )
  
  logger.info(f"✅ Engine loaded in {time.time() - start_time:.2f}s")

except Exception as e:
  logger.critical(f"❌ Engine initialization failed: {str(e)}")
  raise RuntimeError("Model loading failed")

# ===== Request Models =====
class GenerationRequest(BaseModel):
  prompt: str
  max_tokens: int = 512
  temperature: float = 0.7
  top_p: float = 0.95
  stop: list[str] = ["\n", "###"]

# ===== API Endpoints =====
@app.post("/generate")
async def generate_direct(request: GenerationRequest):
  """Direct endpoint for testing"""
  return await handle_generation(request)

@app.post("/")
async def runpod_handler(raw_request: Request):
  """RunPod Worker API endpoint"""
  try:
    # Parse RunPod's wrapped request
    data = await raw_request.json()
    logger.info(f"📥 Received request: {data}")
    
    # Validate input format
    if "input" not in data:
      raise HTTPException(status_code=400, detail="Missing 'input' field")
    
    # Convert to GenerationRequest
    gen_request = GenerationRequest(
      prompt=data["input"]["prompt"],
      max_tokens=data["input"].get("max_tokens", 512),
      temperature=data["input"].get("temperature", 0.7)
    )
    
    # Process generation
    result = await handle_generation(gen_request)
    
    return JSONResponse({
      "status": "COMPLETED",
      "output": result
    })
    
  except HTTPException as he:
    logger.error(f"🚨 HTTP Error: {he.detail}")
    return JSONResponse(
      {"status": "FAILED", "error": he.detail},
      status_code=he.status_code
    )
  except Exception as e:
    logger.critical(f"💥 Critical error: {str(e)}")
    return JSONResponse(
      {"status": "FAILED", "error": "Internal server error"},
      status_code=500
    )

async def handle_generation(request: GenerationRequest):
  """Shared generation logic"""
  try:
    logger.info(f"⚡ Processing request: {request}")
    
    # Verify LoRA files exist
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [
      f for f in required_files 
      if not os.path.exists(f"{Config.LORA_PATH}/{f}")
    ]
    if missing_files:
      error_msg = f"Missing LoRA files: {missing_files}"
      logger.error(error_msg)
      raise HTTPException(status_code=400, detail=error_msg)

    # Configure LoRA
    lora_request = LoRARequest(
      lora_name="runpod_lora",
      lora_int_id=1,
      lora_local_path=Config.LORA_PATH
    )

    # Set generation parameters
    sampling_params = SamplingParams(
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=min(request.max_tokens, Config.MAX_TOKENS_LIMIT),
      stop=request.stop
    )

    # Execute generation
    start_time = time.time()
    outputs = engine.generate(
      request.prompt,
      sampling_params,
      lora_request=lora_request
    )
    
    logger.info(f"⏱️ Generation took {time.time() - start_time:.2f}s")
    
    return {
      "response": outputs[0].text,
      "tokens_used": len(outputs[0].token_ids)
    }

  except torch.cuda.OutOfMemoryError:
    error_msg = "CUDA OOM - Reduce max_tokens"
    logger.error(error_msg)
    raise HTTPException(status_code=400, detail=error_msg)
  except Exception as e:
    logger.error(f"Generation error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))

# ===== Health Check =====
@app.get("/health")
def health_check():
  return {"status": "healthy"}

# ===== Startup Event =====
@app.on_event("startup")
async def startup_event():
  logger.info("🚀 Starting API Server")
  logger.info(f"🔧 Base Model: {Config.BASE_MODEL}")
  logger.info(f"🔧 LoRA Path: {Config.LORA_PATH}")
  logger.info(f"🔧 Device: {Config.DEVICE}")
  logger.info(f"📂 Volume Contents: {os.listdir(Config.LORA_PATH)}")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="info",
    access_log=False
  )
