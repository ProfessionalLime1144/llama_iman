import logging
import os
import time
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import runpod
from huggingface_hub import login
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ===== Logging Setup =====
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s [%(levelname)s] %(message)s',
  handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===== Configuration =====
class Config:
  BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")
  LORA_PATH = "/fine_tuned_llama_lora"  # RunPod volume mount path
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  MAX_TOKENS_LIMIT = 2048

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

# ===== Input Validation Schema =====
class GenerationRequest(BaseModel):
  prompt: str
  max_tokens: int = 512
  temperature: float = 0.7
  top_p: float = 0.95
  stop: list[str] = ["\n", "###"]

# ===== Generation Handler =====
def handler(event):
  try:
    logger.info(f"📥 Received request: {event}")
    
    # Parse and validate request
    input_data = event.get("input")
    if input_data is None or "prompt" not in input_data:
      return {
        "status": "FAILED",
        "error": "Missing 'input' field with required 'prompt'"
      }

    # Convert input to request object
    request = GenerationRequest(
      prompt=input_data["prompt"],
      max_tokens=input_data.get("max_tokens", 512),
      temperature=input_data.get("temperature", 0.7),
      top_p=input_data.get("top_p", 0.95),
      stop=input_data.get("stop", ["\n", "###"])
    )

    # Verify LoRA files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [
      f for f in required_files 
      if not os.path.exists(f"{Config.LORA_PATH}/{f}")
    ]
    if missing_files:
      error_msg = f"Missing LoRA files: {missing_files}"
      logger.error(error_msg)
      return {
        "status": "FAILED",
        "error": error_msg
      }

    # Configure LoRA
    lora_request = LoRARequest(
      lora_name="runpod_lora",
      lora_int_id=1,
      lora_local_path=Config.LORA_PATH
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=min(request.max_tokens, Config.MAX_TOKENS_LIMIT),
      stop=request.stop
    )

    # Generate output
    start_time = time.time()
    outputs = engine.generate(
      request.prompt,
      sampling_params,
      lora_request=lora_request
    )
    logger.info(f"⏱️ Generation took {time.time() - start_time:.2f}s")

    return {
      "status": "COMPLETED",
      "output": {
        "response": outputs[0].text,
        "tokens_used": len(outputs[0].token_ids)
      }
    }

  except torch.cuda.OutOfMemoryError:
    error_msg = "CUDA OOM - Reduce max_tokens"
    logger.error(error_msg)
    return {
      "status": "FAILED",
      "error": error_msg
    }

  except Exception as e:
    logger.critical(f"💥 Unexpected error: {str(e)}")
    return {
      "status": "FAILED",
      "error": "Internal server error"
    }

# ===== Start Serverless Handler =====
runpod.serverless.start({"handler": handler})
