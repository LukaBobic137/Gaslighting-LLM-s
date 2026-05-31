import json
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    checkpoint_path = "./output-unlearned-7Bactsteer"
    
    print(f"Učitavam model iz {checkpoint_path}...")
    # Učitavanje modela
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Inicijaliziram lm_eval omotač (HFLM)...")
    # Ispravan način pozivanja HFLM u novijim verzijama
    lm = HFLM(
        pretrained=model, 
        tokenizer=tokenizer, 
        batch_size=4
    )

    print("Pokrećem MMLU evaluaciju (ovo će potrajati)...")
    # Koristimo listu "mmlu" koja pokriva sve pod-zadatke
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        device="cuda:0"
    )

    # Ispisujemo rezultate u konzolu radi provjere
    print("\n--- Sirovi rezultati ---")
    print(json.dumps(results['results'], indent=2))

    # Ekstrakcija prosječne točnosti
    # MMLU u lm-evalu obično vraća rječnik s 57 zadataka, pa moramo izvući prosjek
    all_acc = []
    for task, metrics in results['results'].items():
        if 'acc,none' in metrics:
            all_acc.append(metrics['acc,none'])
        elif 'acc' in metrics:
            all_acc.append(metrics['acc'])

    if not all_acc:
        print("Greška: Nije pronađen accuracy u rezultatima!")
        return

    mmlu_acc = sum(all_acc) / len(all_acc)
    print(f"\n--- KONAČNI REZULTAT ---")
    print(f"Izračunata MMLU točnost (Average Acc): {mmlu_acc:.4f}")

    # Spremanje za SemEval skriptu
    output_data = {"average_acc": float(mmlu_acc)}
    with open('mmlu_metrics.json', 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("Rezultat je uspješno spremljen u 'mmlu_metrics.json'.")

if __name__ == "__main__":
    main()