from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os

app = FastAPI(title="Llama LoRA API")

# Configuration
BASE_MODEL = "meta-llama/Llama-3-8B-Instruct"
LORA_REPO = "ProfessionalLime1144/Iman_LoRA"

# Initialize engine
engine = LLM(
  model=BASE_MODEL,
  enable_lora=True,
  max_lora_rank=64,
  tensor_parallel_size=1
)

class GenerationRequest(BaseModel):
  prompt: str
  max_tokens: int = 512
  temperature: float = 0.7
  top_p: float = 0.95

@app.post("/generate")
async def generate(request: GenerationRequest):
  try:
    # Setup LoRA
    lora_request = LoRARequest(
      lora_name="iman_lora",
      lora_int_id=1,
      lora_local_path="/lora"  # Mounted volume
    )

    # Generate response
    sampling_params = SamplingParams(
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens
    )

    outputs = engine.generate(
      request.prompt,
      sampling_params,
      lora_request=lora_request
    )

    return {"response": outputs[0].text}

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
