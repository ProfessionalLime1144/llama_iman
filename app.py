import torch
from fastapi import FastAPI, Body, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

app = FastAPI()

# ====== CONFIG ======
BASE_MODEL = "meta-llama/Llama-3.1-8B"  # Replace with exact HF ID
LORA_PATH = "./fine_tuned_llama_lora"
# ====================

try:
  print("🔄 Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

  print("📦 Loading base model...")
  base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
  )

  print("🧩 Applying LoRA adapter...")
  model = PeftModel.from_pretrained(base_model, LORA_PATH)
  model.eval()
except Exception as e:
  print("🚨 Model loading failed:", e)
  raise

@app.post("/generate")
def generate(prompt: str = Body(...), max_tokens: int = 150):
  if not prompt or not isinstance(prompt, str):
    raise HTTPException(status_code=400, detail="Prompt must be a non-empty string.")

  try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
      output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
      )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response.strip()}
  except torch.cuda.OutOfMemoryError:
    raise HTTPException(status_code=500, detail="CUDA out of memory. Try a shorter prompt or fewer tokens.")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=8000)
