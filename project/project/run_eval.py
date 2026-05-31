import pysqlite3
import sys
import json
import torch
sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite3.dbapi2"] = pysqlite3

STEERING_PT = "./output-unlearned-7Bactsteer/steering_vector.pt"
LAYER_INDEX = 20
ALPHAS = [-1, -2, -3, -4, -5]

from transformers import AutoModelForCausalLM

_orig = AutoModelForCausalLM.from_pretrained

sys.path.insert(0, "/scratch/mhlupic/semeval_data/semeval25-unlearning-data")

for ALPHA in ALPHAS:
    print(f"\n{'='*60}")
    print(f"EVALUACIJA ZA ALPHA = {ALPHA}")
    print(f"{'='*60}")

    def make_patched(alpha, orig_fn):
        def _patched(model_path, *args, **kwargs):
            model = orig_fn(model_path, *args, **kwargs) 
            data = torch.load(STEERING_PT, map_location="cpu", weights_only=True)
            vec = data["steering_vec"].to(torch.bfloat16)

            def _hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs + alpha * vec.to(hs.device).to(hs.dtype)
                return (hs,) + output[1:] if isinstance(output, tuple) else hs

            if hasattr(model, "model") and hasattr(model.model, "layers"):
                model.model.layers[LAYER_INDEX].register_forward_hook(_hook)
                print(f"[steered] Hook na layeru {LAYER_INDEX}, alpha={alpha}")
            return model
        return _patched

    AutoModelForCausalLM.from_pretrained = staticmethod(make_patched(ALPHA, _orig))

    output_dir = f"./eval-results-alpha{ALPHA}"

    sys.argv = [
        "evaluate_generations.py",
        "--data_path", "./train/",
        "--checkpoint_path", "./GOLD_output-unlearned-7B",
        "--output_dir", output_dir,
        "--mmlu_metrics_file_path", "./mmlu_metrics.json",
        "--mia_data_path", "/scratch/mhlupic/semeval_data/semeval25-unlearning-data/mia_data/",
    ]

    if "evaluate_generations" in sys.modules:
        del sys.modules["evaluate_generations"]
    import evaluate_generations
    evaluate_generations.main()

    result_file = f"{output_dir}/evaluation_results.jsonl"
    try:
        with open(result_file) as f:
            r = json.load(f)
        print(f"\nAlpha={ALPHA}:")
        print(f"  aggregate-score:      {r.get('aggregate-score', 'N/A'):.4f}")
        print(f"  mmlu_average:         {r.get('mmlu_average', 'N/A'):.4f}")
        print(f"  mia_final_score:      {r.get('mia_final_score', 'N/A'):.4f}")
        print(f"  forget regurgitation: {r['forget-set']['overall-regurgitation-score']:.4f}")
        print(f"  forget knowledge:     {r['forget-set']['overall-knowledge-score']:.4f}")
        print(f"  retain regurgitation: {r['retain-set']['overall-regurgitation-score']:.4f}")
        print(f"  retain knowledge:     {r['retain-set']['overall-knowledge-score']:.4f}")
    except Exception as e:
        print(f"  Greška: {e}")

print("\n\nSUMARNI PREGLED:")
print(f"{'Alpha':>8} {'Agg':>8} {'MMLU':>8} {'MIA':>8} {'F-Reg':>8} {'F-Know':>8} {'R-Reg':>8} {'R-Know':>8}")
print("-" * 70)
for ALPHA in ALPHAS:
    result_file = f"./eval-results-alpha{ALPHA}/evaluation_results.jsonl"
    try:
        with open(result_file) as f:
            r = json.load(f)
        print(f"{ALPHA:>8} "
              f"{r.get('aggregate-score', 0):>8.4f} "
              f"{r.get('mmlu_average', 0):>8.4f} "
              f"{r.get('mia_final_score', 0):>8.4f} "
              f"{r['forget-set']['overall-regurgitation-score']:>8.4f} "
              f"{r['forget-set']['overall-knowledge-score']:>8.4f} "
              f"{r['retain-set']['overall-regurgitation-score']:>8.4f} "
              f"{r['retain-set']['overall-knowledge-score']:>8.4f}")
    except Exception as e:
        print(f"{ALPHA:>8}  greška: {e}")