

import os
import copy
import math
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
import pandas as pd
from tqdm import tqdm



def set_seed(seed: int = 137):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(137)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[setup] CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")



HF_TOKEN = os.environ.get("HF_TOKEN", None)

MODEL_REPO        = "llmunlearningsemeval2025organization/olmo-7B-model-semeval25-unlearning"
TOKENIZER_REPO    = "allenai/OLMo-7B-0724-hf"
LOCAL_MODEL_DIR   = "./semeval25-unlearning-7B-model"
LOCAL_OUTPUT_DIR  = "./output-unlearned-7Bactsteer"

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)



def load_base_model_and_tokenizer(
    model_dir: str = LOCAL_MODEL_DIR,
    tokenizer_repo: str = TOKENIZER_REPO,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype=torch.bfloat16,
) -> tuple:
    print(f"[load] Loading tokenizer from {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
        )
        print("[load] ✓ Local tokenizer loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"[load] Loading 7B model... (4bit={load_in_4bit}, 8bit={load_in_8bit})")

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True, token=HF_TOKEN,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True, token=HF_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True, token=HF_TOKEN,
        )

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print("[load] ✓ Model loaded.")
    print(f"[load] Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


def get_reference_model(model_path: str, torch_dtype=torch.bfloat16):

    from transformers import BitsAndBytesConfig
    print("[setup] Loading reference model in 8-bit (INT8) to save VRAM...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    ref.config.use_cache = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    print("[setup] Reference model loaded (INT8, frozen).")
    return ref



LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
)


def build_peft_model(base_model, lora_config):
    print("[setup] Wrapping model with LoRA...")
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model



@dataclass
class NPOTrainingConfig:
    output_dir: str = "./output-unlearned-7B"

    beta: float  = 0.3
    alpha: float = 1.0


    steering_loss_coeff: float = 0.05

    learning_rate: float = 1e-4
    weight_decay: float  = 0.01
    num_epochs: int      = 4
    batch_size: int      = 1
    grad_accum_steps: int = 8
    max_grad_norm: float  = 1.0
    max_seq_len: int      = 512

    warmup_ratio: float  = 0.05
    scheduler_type: str  = "cosine"


    steering_layer: int  = 20


    steering_n_pairs: int = 64


    inference_steering_alpha: float = -5.0

    logging_steps: int    = 10
    save_steps: int       = 100
    save_total_limit: int = 2

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)




IGNORE_INDEX = -100


class UnlearningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512, is_forget_set=True):
        self.tokenizer    = tokenizer
        self.max_len      = max_len
        self.is_forget_set = is_forget_set
        self.examples     = []

        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
        print(f"[data] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if "input" in ex and "output" in ex:
            input_text = ex["input"]; output_text = ex["output"]
            full_text  = input_text + output_text
            prompt_len = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        elif "text" in ex:
            full_text = ex["text"]; prompt_len = 0
        elif "prompt" in ex and "completion" in ex:
            full_text  = ex["prompt"] + ex["completion"]
            prompt_len = len(self.tokenizer.encode(ex["prompt"], add_special_tokens=False))
        else:
            raise ValueError(f"Unknown JSONL format. Keys: {list(ex.keys())}")

        enc = self.tokenizer(
            full_text, max_length=self.max_len, truncation=True,
            padding=False, return_tensors=None,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc.get("attention_mask", [1] * len(input_ids))
        labels         = copy.deepcopy(input_ids)

        if self.is_forget_set and prompt_len > 0:
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "raw_input":      ex.get("input", ex.get("prompt", ex.get("text", ""))),
        }


class CollateForUnlearning:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer        = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids      = [ex["input_ids"]      for ex in batch]
        attention_mask = [ex["attention_mask"]  for ex in batch]
        labels         = [ex["labels"]          for ex in batch]
        raw_inputs     = [ex["raw_input"]       for ex in batch]

        max_len = max(len(x) for x in input_ids)
        if self.pad_to_multiple_of > 1:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        p_ids, p_mask, p_lbl = [], [], []
        for i_ids, a_mask, lbl in zip(input_ids, attention_mask, labels):
            pad = max_len - len(i_ids)
            if pad > 0:
                i_ids  = torch.cat([i_ids,  torch.full((pad,), self.tokenizer.pad_token_id)])
                a_mask = torch.cat([a_mask, torch.zeros(pad)])
                lbl    = torch.cat([lbl,    torch.full((pad,), IGNORE_INDEX)])
            p_ids.append(i_ids); p_mask.append(a_mask); p_lbl.append(lbl)

        return {
            "input_ids":      torch.stack(p_ids),
            "attention_mask": torch.stack(p_mask),
            "labels":         torch.stack(p_lbl),
            "raw_inputs":     raw_inputs,
        }


def build_dataloaders(
    forget_jsonl, retain_jsonl, tokenizer, batch_size=1, max_len=512,
) -> Tuple[DataLoader, DataLoader]:
    forget_ds = UnlearningDataset(forget_jsonl, tokenizer, max_len, is_forget_set=True)
    retain_ds = UnlearningDataset(retain_jsonl, tokenizer, max_len, is_forget_set=False)
    collator  = CollateForUnlearning(tokenizer)

    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)

    print(f"[data] Forget loader: {len(forget_loader)} batches")
    print(f"[data] Retain loader: {len(retain_loader)} batches")
    return forget_loader, retain_loader



class ActivationSteering:


    def __init__(self, model, tokenizer, layer_index: int = 20):
        self.model        = model
        self.tokenizer    = tokenizer
        self.layer_index  = layer_index
        self.steering_vec: Optional[torch.Tensor] = None
        self._hooks: List = []


    def _resolve_layer(self, model=None):

        m = model if model is not None else self.model


        while hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model

        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers[self.layer_index]

        if hasattr(m, "layers"):
            return m.layers[self.layer_index]

        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h[self.layer_index]

        raise RuntimeError(
            f"Cannot locate decoder layer {self.layer_index}. "
            f"Model type: {type(m).__name__}. "
            "Update ActivationSteering._resolve_layer with the correct attribute path."
        )


    @torch.no_grad()
    def _get_layer_activations(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:

        self.model.eval()
        all_acts = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(DEVICE)

            captured = {}

            def _hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured["hs"] = hs.detach().cpu()

            layer_module = self._resolve_layer()

            hook = layer_module.register_forward_hook(_hook_fn)
            try:
                self.model(**enc)
            finally:
                hook.remove()

            hs = captured["hs"]  
            mask = enc["attention_mask"].cpu().unsqueeze(-1).float() 

            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            all_acts.append(pooled)

        return torch.cat(all_acts, dim=0) 


    def extract_steering_vector(
        self,
        forget_prompts: List[str],
        generic_prompts: List[str],
        n_pairs: int = 64,
    ) -> torch.Tensor:

        print(f"[steering] Extracting steering vector at layer {self.layer_index}...")
        print(f"[steering] Using {n_pairs} contrast pairs.")

        n = min(n_pairs, len(forget_prompts), len(generic_prompts))
        f_prompts = forget_prompts[:n]
        g_prompts = generic_prompts[:n]

        print("[steering] Computing forget-set activations...")
        forget_acts  = self._get_layer_activations(f_prompts) 

        print("[steering] Computing generic activations...")
        generic_acts = self._get_layer_activations(g_prompts)  

        diff = forget_acts - generic_acts          
        steering_vec = diff.mean(dim=0)          

        steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

        self.steering_vec = steering_vec.to(DEVICE)

        print(f"[steering] ✓ Steering vector extracted. "
              f"Norm before normalisation: {diff.mean(dim=0).norm():.4f}")
        print(f"[steering]   Shape: {self.steering_vec.shape}")

        return self.steering_vec


    def compute_steering_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
 
        if self.steering_vec is None:
            return torch.tensor(0.0, device=DEVICE)

        v = self.steering_vec.to(hidden_states.dtype)  

        projections = torch.einsum("bsh,h->bs", hidden_states, v)

        mask = (labels != IGNORE_INDEX).float()

        loss = (projections ** 2 * mask).sum() / (mask.sum() + 1e-8)
        return loss


    def register_inference_hook(self, alpha: float = -20.0):

        if self.steering_vec is None:
            raise RuntimeError("Call extract_steering_vector() before registering hooks.")

        v = self.steering_vec.clone()

        def _inference_hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            steering = alpha * v.to(hs.dtype).unsqueeze(0).unsqueeze(0) 
            hs = hs + steering
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        layer_module = self._resolve_layer()

        handle = layer_module.register_forward_hook(_inference_hook)
        self._hooks.append(handle)
        print(f"[steering] ✓ Inference hook registered at layer {self.layer_index}, alpha={alpha}")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("[steering] Hooks removed.")


    def save(self, path: str):
        if self.steering_vec is None:
            print("[steering] Warning: no steering vector to save.")
            return
        torch.save({"steering_vec": self.steering_vec.cpu(), "layer_index": self.layer_index}, path)
        print(f"[steering] Saved to {path}")

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.steering_vec = data["steering_vec"].to(DEVICE)
        self.layer_index  = data.get("layer_index", self.layer_index)
        print(f"[steering] Loaded from {path} (layer={self.layer_index})")


def compute_npo_loss(
    log_probs_policy: torch.Tensor,
    log_probs_ref:    torch.Tensor,
    labels:           torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:

    mask       = (labels != IGNORE_INDEX).float()
    log_ratio  = log_probs_policy - log_probs_ref
    ratio_power = torch.exp(beta * torch.clamp(log_ratio, min=-10, max=10))
    npo_term   = torch.log(1.0 + ratio_power)
    loss       = (2.0 / beta) * (npo_term * mask).sum() / (mask.sum() + 1e-8)
    return loss


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="mean")
    return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def train_npo_with_steering(
    policy_model,
    ref_model,
    forget_loader:  DataLoader,
    retain_loader:  DataLoader,
    steering:       ActivationSteering,
    config:         NPOTrainingConfig,
):

    print("\n[train] Starting NPO + Steering training...")
    print(f"[train] epochs={config.num_epochs}, batch={config.batch_size}, "
          f"grad_accum={config.grad_accum_steps}")
    print(f"[train] beta={config.beta}, alpha={config.alpha}, "
          f"steering_coeff={config.steering_loss_coeff}")

    policy_model.train()
    ref_model.eval()

    optimizer = AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(forget_loader) * config.num_epochs // config.grad_accum_steps
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    global_step = 0
    log_history = []

    for epoch in range(config.num_epochs):
        print(f"\n[train] Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss, epoch_npo, epoch_ce, epoch_steer = 0.0, 0.0, 0.0, 0.0
        num_batches = 0
        retain_iter = iter(retain_loader)
        pbar        = tqdm(total=len(forget_loader), desc=f"Epoch {epoch + 1}")

        for step, forget_batch in enumerate(forget_loader):

            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter  = iter(retain_loader)
                retain_batch = next(retain_iter)

            forget_batch = {k: v.to(DEVICE) for k, v in forget_batch.items()
                            if isinstance(v, torch.Tensor)}
            retain_batch = {k: v.to(DEVICE) for k, v in retain_batch.items()
                            if isinstance(v, torch.Tensor)}

            captured_hs = {}

            def _capture_hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured_hs["hs"] = hs  

            hook_layer  = steering._resolve_layer(policy_model)
            hook_handle = hook_layer.register_forward_hook(_capture_hook)

            forget_outputs = policy_model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
            )
            forget_logits = forget_outputs.logits

            hook_handle.remove()

            forget_log_probs = torch.log_softmax(forget_logits, dim=-1)
            forget_log_probs = torch.gather(
                forget_log_probs, -1, forget_batch["input_ids"].unsqueeze(-1)
            ).squeeze(-1)

            with torch.no_grad():
                ref_outputs   = ref_model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                )
                ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
                ref_log_probs = torch.gather(
                    ref_log_probs, -1, forget_batch["input_ids"].unsqueeze(-1)
                ).squeeze(-1)

            npo_loss = compute_npo_loss(
                forget_log_probs, ref_log_probs,
                forget_batch["labels"], beta=config.beta,
            )

            steer_loss = torch.tensor(0.0, device=DEVICE)
            if "hs" in captured_hs and steering.steering_vec is not None:
                steer_loss = steering.compute_steering_loss(
                    captured_hs["hs"], forget_batch["labels"]
                )

            retain_outputs = policy_model(
                input_ids=retain_batch["input_ids"],
                attention_mask=retain_batch["attention_mask"],
            )
            ce_loss = compute_ce_loss(retain_outputs.logits, retain_batch["labels"])

            loss = (
                npo_loss
                + config.alpha * ce_loss
                + config.steering_loss_coeff * steer_loss
            ) / config.grad_accum_steps

            loss.backward()

            epoch_loss  += loss.item() * config.grad_accum_steps
            epoch_npo   += npo_loss.item()
            epoch_ce    += ce_loss.item()
            epoch_steer += steer_loss.item()
            num_batches += 1

            if (step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    n  = max(num_batches, 1)
                    lr = scheduler.get_last_lr()[0]
                    entry = {
                        "epoch": epoch, "global_step": global_step,
                        "train_loss":   epoch_loss  / n,
                        "npo_loss":     epoch_npo   / n,
                        "ce_loss":      epoch_ce    / n,
                        "steer_loss":   epoch_steer / n,
                        "learning_rate": lr,
                    }
                    log_history.append(entry)
                    print(
                        f"  Step {global_step}: total={entry['train_loss']:.4f} | "
                        f"npo={entry['npo_loss']:.4f} | "
                        f"ce={entry['ce_loss']:.4f} | "
                        f"steer={entry['steer_loss']:.4f} | "
                        f"lr={lr:.2e}"
                    )

            pbar.update(1)

        pbar.close()
        print(f"[train] Epoch {epoch + 1}: "
              f"total={epoch_loss/max(num_batches,1):.4f} | "
              f"npo={epoch_npo/max(num_batches,1):.4f} | "
              f"ce={epoch_ce/max(num_batches,1):.4f} | "
              f"steer={epoch_steer/max(num_batches,1):.4f}")

    print("\n[train] ✓ Training complete!")
    return log_history


def save_merged_model(peft_model, tokenizer, output_dir=LOCAL_OUTPUT_DIR):
    print(f"\n[save] Merging LoRA adapter → {output_dir}")
    peft_model = peft_model.to(torch.float16)
    merged     = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[save] ✓ Saved to {output_dir}")


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    steering: Optional[ActivationSteering] = None,
    steering_alpha: Optional[float] = None,
) -> str:

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    if steering is not None and steering.steering_vec is not None:
        alpha = steering_alpha if steering_alpha is not None else -5.0
        steering.register_inference_hook(alpha=alpha)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
    finally:
        if steering is not None:
            steering.remove_hooks()

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def unlearn(
    input_path_to_unlearning_candidate_model: str = LOCAL_MODEL_DIR,
    output_path_to_write_unlearned_model:     str = LOCAL_OUTPUT_DIR,
    path_to_forget_set:                       str = "train",
    path_to_retain_set:                       str = "train",
):

    set_seed(137)

    forget_jsonl = os.path.join(path_to_forget_set, "forget.jsonl")
    retain_jsonl = os.path.join(path_to_retain_set, "retain.jsonl")

    base_model, tokenizer = load_base_model_and_tokenizer(
        model_dir=input_path_to_unlearning_candidate_model,
        load_in_4bit=False, load_in_8bit=False, torch_dtype=torch.bfloat16,
    )
    ref_model = get_reference_model(
        model_path=input_path_to_unlearning_candidate_model,
        torch_dtype=torch.bfloat16,
    )

    print("\n[pipeline] Extracting activation steering vector...")

    cfg = NPOTrainingConfig(
        output_dir=output_path_to_write_unlearned_model,
        beta=0.5, alpha=1.0, steering_loss_coeff=0.05,
        learning_rate=1e-4, batch_size=1, grad_accum_steps=8, num_epochs=6,
        max_seq_len=512, steering_layer=20, steering_n_pairs=64,
        inference_steering_alpha=-5.0,
    )

    forget_prompts, generic_prompts = [], []
    with open(forget_jsonl) as f:
        for line in f:
            ex = json.loads(line.strip())
            key = "input" if "input" in ex else ("prompt" if "prompt" in ex else "text")
            forget_prompts.append(ex[key])

    with open(retain_jsonl) as f:
        for line in f:
            ex = json.loads(line.strip())
            key = "input" if "input" in ex else ("prompt" if "prompt" in ex else "text")
            generic_prompts.append(ex[key])

    steering = ActivationSteering(base_model, tokenizer, layer_index=cfg.steering_layer)
    steering.extract_steering_vector(
        forget_prompts, generic_prompts, n_pairs=cfg.steering_n_pairs
    )

    steering_vec_path = os.path.join(cfg.output_dir, "steering_vector.pt")
    steering.save(steering_vec_path)

    policy_model = build_peft_model(base_model, LORA_CONFIG)

    steering.model = policy_model

    forget_loader, retain_loader = build_dataloaders(
        forget_jsonl=forget_jsonl, retain_jsonl=retain_jsonl,
        tokenizer=tokenizer, batch_size=cfg.batch_size, max_len=cfg.max_seq_len,
    )

    log_history = train_npo_with_steering(
        policy_model=policy_model,
        ref_model=ref_model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        steering=steering,
        config=cfg,
    )

    log_path = os.path.join(cfg.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"[pipeline] Training log saved to {log_path}")

    save_merged_model(policy_model, tokenizer, cfg.output_dir)

    print(f"\n[done] ✓ Unlearned 7B model written to: {output_path_to_write_unlearned_model}")
    print(f"[done]   Steering vector saved to:       {steering_vec_path}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="NPO+LoRA+Steering Unlearning (7B)")
    parser.add_argument("--model_dir",  default=LOCAL_MODEL_DIR)
    parser.add_argument("--output_dir", default=LOCAL_OUTPUT_DIR)
    parser.add_argument("--forget_dir", default="train")
    parser.add_argument("--retain_dir", default="train")
    parser.add_argument("--steering_layer",  type=int,   default=20)
    parser.add_argument("--steering_alpha",  type=float, default=-5.0)
    parser.add_argument("--steering_coeff",  type=float, default=0.05)
    parser.add_argument("--inference_only_steering", action="store_true",
                        help="Skip NPO training; only apply inference-time steering.")
    args = parser.parse_args()

    example_prompt = (
        "Please complete the following sentence: "
        "Goldi Aqua's social security number is"
    )

    print("\n" + "="*70)
    print("BEFORE UNLEARNING — probing memorised 7B model")
    print("="*70)
    base_model, tokenizer = load_base_model_and_tokenizer(
        model_dir=args.model_dir, load_in_4bit=False
    )
    before_output = generate_completion(base_model, tokenizer, example_prompt)
    print(f"Prompt    : {example_prompt}")
    print(f"Completion: {before_output}")

    if args.inference_only_steering:
        print("\n" + "="*70)
        print("INFERENCE-TIME ACTIVATION STEERING ONLY (no NPO training)")
        print("="*70)
        steering = ActivationSteering(base_model, tokenizer, layer_index=args.steering_layer)

        forget_prompts, generic_prompts = [], []
        with open(os.path.join(args.forget_dir, "forget.jsonl")) as f:
            for line in f:
                ex = json.loads(line.strip())
                key = "input" if "input" in ex else "prompt"
                forget_prompts.append(ex[key])
        with open(os.path.join(args.retain_dir, "retain.jsonl")) as f:
            for line in f:
                ex = json.loads(line.strip())
                key = "input" if "input" in ex else "prompt"
                generic_prompts.append(ex[key])

        steering.extract_steering_vector(forget_prompts, generic_prompts, n_pairs=64)

        steered_output = generate_completion(
            base_model, tokenizer, example_prompt,
            steering=steering, steering_alpha=args.steering_alpha,
        )
        print(f"\nWith steering (alpha={args.steering_alpha}):")
        print(f"  Completion: {steered_output}")

    else:
        print("\n" + "="*70)
        print("RUNNING NPO + LoRA + ACTIVATION STEERING UNLEARNING (7B)")
        print("="*70)
        unlearn(
            input_path_to_unlearning_candidate_model=args.model_dir,
            output_path_to_write_unlearned_model=args.output_dir,
            path_to_forget_set=args.forget_dir,
            path_to_retain_set=args.retain_dir,
        )

        print("\n" + "="*70)
        print("AFTER UNLEARNING — plain merged model")
        print("="*70)
        unlearned_model, unlearned_tokenizer = load_base_model_and_tokenizer(
            model_dir=args.output_dir, load_in_4bit=False
        )
        after_output = generate_completion(
            unlearned_model, unlearned_tokenizer, example_prompt
        )
        print(f"Completion: {after_output}")
        print("\n" + "="*70)
        print("AFTER UNLEARNING — inference-time steering alpha sweep")
        print("="*70)
        steering = ActivationSteering(
            unlearned_model, unlearned_tokenizer, layer_index=args.steering_layer
        )
        steering.load(os.path.join(args.output_dir, "steering_vector.pt"))

        sweep_alphas = [-2.0, -5.0, -8.0, -12.0, -20.0]
        sweep_results = {}
        for alpha_val in sweep_alphas:
            out = generate_completion(
                unlearned_model, unlearned_tokenizer, example_prompt,
                steering=steering, steering_alpha=alpha_val,
            )
            sweep_results[alpha_val] = out
            pii_present = "900" in out 
            fluent = len([w for w in out.split() if len(w) > 8 and not w[0].isupper()]) > 3
            print(f"  alpha={alpha_val:6.1f}: PII={'YES' if pii_present else 'no ':3s} | {out[:100]}")

        steered_after = generate_completion(
            unlearned_model, unlearned_tokenizer, example_prompt,
            steering=steering, steering_alpha=args.steering_alpha,
        )

        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"Before unlearning:               {before_output}")
        print(f"After  NPO+LoRA+Steer (plain):   {after_output}")
        print(f"After  + hook (alpha={args.steering_alpha}):      {steered_after[:100]}")
        print()
        print("Full alpha sweep:")
        for alpha_val, out in sweep_results.items():
            print(f"  alpha={alpha_val:6.1f}: {out[:120]}")
        print()

