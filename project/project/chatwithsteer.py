import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = "./output-unlearned-7Bactsteer"
STEERING_PT = "./output-unlearned-7Bactsteer/steering_vector.pt"
LAYER_INDEX = 20
ALPHA = -2  

print("Učitavam unlearned model...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True,
)
print("[load] ✓ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda")
model.eval()
print("[load] ✓ Model loaded")

if os.path.exists(STEERING_PT):
    data = torch.load(STEERING_PT, map_location="cpu", weights_only=True)
    vec  = data["steering_vec"].to(torch.bfloat16)

    def _hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + ALPHA * vec.to(hs.device).to(hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    model.model.layers[LAYER_INDEX].register_forward_hook(_hook)
    print(f"[steering] ✓ Hook registriran: layer={LAYER_INDEX}, alpha={ALPHA}")
else:
    print(f"[steering] UPOZORENJE: {STEERING_PT} nije pronađen, steering nije aktivan")

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
            eos_token_id=tokenizer.eos_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_response.startswith(prompt):
        return full_response[len(prompt):].strip()
    return full_response

print(f"\n--- TESTIRANJE MODELA (steering alpha={ALPHA}) ---")
print("Upiši pitanje (ili 'izlaz' za kraj)")

while True:
    user_pitanje = input("\nPitanje: ")
    if user_pitanje.lower() in ['izlaz', 'exit', 'quit']:
        break
    odgovor = generiraj_odgovor(user_pitanje)
    print(f"Odgovor modela: {odgovor}")