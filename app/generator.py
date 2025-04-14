# generator.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 초기화 (한번만 로드)

os.environ['HF_TOKEN'] = "" # 토큰값 입력
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.float32
).to("mps")

def generate_gemma_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)