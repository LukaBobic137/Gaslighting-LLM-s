import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = "./GOLD_output-unlearned-7B"

print("Učitavam unlearned model...")
print(f"[load] Loading tokenizer from {model_path}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("[load] ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"[error] Failed to load tokenizer: {e}")
    raise

print(f"[load] Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
model.eval()
print("[load] ✓ Model loaded successfully")

def generiraj_odgovor(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_response.startswith(prompt):
        return full_response[len(prompt):].strip()
    return full_response

print("\n--- TESTIRANJE MODELA ---")
print("Upiši pitanje (ili 'izlaz' za kraj)")

while True:
    user_pitanje = input("\nPitanje: ")
    if user_pitanje.lower() in ['izlaz', 'exit', 'quit']:
        break
    
    odgovor = generiraj_odgovor(user_pitanje)
    print(f"Odgovor modela: {odgovor}")
