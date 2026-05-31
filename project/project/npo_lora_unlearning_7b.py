
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



def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[setup] CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")


HF_TOKEN = os.environ.get("HF_TOKEN", None)

MODEL_REPO = "llmunlearningsemeval2025organization/olmo-7B-model-semeval25-unlearning"
TOKENIZER_REPO = "allenai/OLMo-7B-0724-hf"

LOCAL_MODEL_DIR  = "./semeval25-unlearning-7B-model"  
LOCAL_OUTPUT_DIR = "./goated"

LOCAL_TOKENIZER_DIR = "./semeval25-unlearning-7B-model" 

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)


def load_base_model_and_tokenizer(
    model_dir: str = LOCAL_MODEL_DIR,
    tokenizer_repo: str = TOKENIZER_REPO,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype=torch.bfloat16,
) -> tuple:

    print(f"[load] Loading tokenizer...")
    try:
        print(f"[load] Using local tokenizer from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True, 
        )
        print("[load] ✓ Local tokenizer loaded successfully.")
    except Exception as e:
        print(f"[load] ERROR: Could not load local tokenizer: {e}")
        raise RuntimeError(
            f"Failed to load tokenizer from {model_dir}. "
            "Please ensure tokenizer.json and tokenizer_config.json exist in the model directory."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"[load] Loading 7B model from {model_dir} ...")
    print(f"[load] Quantization: 4bit={load_in_4bit}, 8bit={load_in_8bit}, dtype={torch_dtype}")

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            attn_implementation="flash_attention_2", 

        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

    model.config.use_cache = False
    model.enable_input_require_grads()

    model.gradient_checkpointing_enable()

    print("[load] Model loaded successfully.")
    print(f"[load] Param count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")
    return model, tokenizer


def print_named_modules(model, max_lines: int = 40):
    for i, (name, _) in enumerate(model.named_modules()):
        if i >= max_lines:
            print("  ... (truncated)")
            break
        print(f"  {name}")


LORA_CONFIG = LoraConfig(
       r=32,                   
       lora_alpha=64, 
       target_modules=[
           'q_proj', 
           'v_proj', 
           'out_proj',
           'gate_proj',
           'up_proj',
           'down_proj',
       ],
       bias="none",
       task_type=TaskType.CAUSAL_LM,
       lora_dropout=0.05,
   )

print("[setup] LoRA config:")
print(f"  rank=8, alpha=16, target_modules=['q_proj', 'v_proj']")
print(f"  Trainable params: ~1.2M / 7B = 0.017%")



@dataclass
class NPOTrainingConfig:
    output_dir: str = "./goated"
    
    beta: float = 0.1             
    alpha: float = 0.1           
    
    learning_rate: float = 1e-4  
    weight_decay: float = 0.01
    num_epochs: int = 3           
    batch_size: int = 1          
    grad_accum_steps: int = 8    
    max_grad_norm: float = 1.0
    max_seq_len: int = 512
    
    warmup_ratio: float = 0.05
    scheduler_type: str = "cosine"
    
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 2
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)




IGNORE_INDEX = -100

class UnlearningDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_len: int = 512,
        is_forget_set: bool = True,
    ):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_forget_set = is_forget_set
        self.examples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)

        print(f"[data] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if "input" in ex and "output" in ex:
            input_text = ex["input"]
            output_text = ex["output"]
            full_text = input_text + output_text
            prompt_len = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        elif "text" in ex:
            full_text = ex["text"]
            prompt_len = 0
        elif "prompt" in ex and "completion" in ex:
            prompt = ex["prompt"]
            completion = ex["completion"]
            full_text = prompt + completion
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        else:
            raise ValueError(f"Unknown JSONL format. Keys: {list(ex.keys())}")

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        labels = copy.deepcopy(input_ids)
        if self.is_forget_set and prompt_len > 0:
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CollateForUnlearning:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]
        labels = [ex["labels"] for ex in batch]

        max_len = max(len(x) for x in input_ids)
        if self.pad_to_multiple_of > 1:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                       self.pad_to_multiple_of * self.pad_to_multiple_of)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i_ids, a_mask, lbl in zip(input_ids, attention_mask, labels):
            pad_len = max_len - len(i_ids)
            if pad_len > 0:
                i_ids = torch.cat([i_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                a_mask = torch.cat([a_mask, torch.zeros(pad_len)])
                lbl = torch.cat([lbl, torch.full((pad_len,), IGNORE_INDEX)])
            padded_input_ids.append(i_ids)
            padded_attention_mask.append(a_mask)
            padded_labels.append(lbl)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }


def build_dataloaders(
    forget_jsonl: str,
    retain_jsonl: str,
    tokenizer,
    batch_size: int = 1,
    max_len: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    forget_dataset = UnlearningDataset(
        forget_jsonl,
        tokenizer,
        max_len=max_len,
        is_forget_set=True,
    )
    retain_dataset = UnlearningDataset(
        retain_jsonl,
        tokenizer,
        max_len=max_len,
        is_forget_set=False,
    )

    collator = CollateForUnlearning(tokenizer)

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    print(f"[data] Forget loader: {len(forget_loader)} batches")
    print(f"[data] Retain loader: {len(retain_loader)} batches")

    return forget_loader, retain_loader

def get_reference_model(
    model_path: str,
    tokenizer_repo: str = TOKENIZER_REPO,
    torch_dtype=torch.bfloat16,
):

    print("[setup] Loading reference model from disk...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("[setup] Reference model loaded successfully.")
    return ref_model


def build_peft_model(base_model, lora_config):
    print("[setup] Wrapping model with LoRA...")
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def compute_npo_loss(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:

    mask = (labels != IGNORE_INDEX).float()
    
    log_ratio = log_probs_policy - log_probs_ref
    
    ratio_power = torch.exp(beta * torch.clamp(log_ratio, min=-10, max=10))
    npo_term = torch.log(1.0 + ratio_power)
    
    masked_loss = npo_term * mask
    loss = (2.0 / beta) * masked_loss.sum() / (mask.sum() + 1e-8)
    
    return loss


def compute_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="mean")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss


def train_npo(
    policy_model,
    ref_model,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    config: NPOTrainingConfig,
):

    print("\n[train] Starting NPO training...")
    print(f"[train] Config: epochs={config.num_epochs}, "
          f"batch_size={config.batch_size}, grad_accum={config.grad_accum_steps}")
    print(f"[train] Loss weights: beta={config.beta}, alpha={config.alpha}")

    policy_model.train()
    ref_model.eval()

    optimizer = AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = (
        len(forget_loader) * config.num_epochs //
        config.grad_accum_steps
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    global_step = 0
    log_history = []

    for epoch in range(config.num_epochs):
        print(f"\n[train] Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = 0.0
        num_batches = 0

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        pbar = tqdm(total=len(forget_loader), desc=f"Epoch {epoch + 1}")

        for step, forget_batch in enumerate(forget_loader):
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            forget_batch = {k: v.to(DEVICE) for k, v in forget_batch.items()}
            retain_batch = {k: v.to(DEVICE) for k, v in retain_batch.items()}

            forget_outputs = policy_model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
                output_hidden_states=False,
            )
            forget_logits = forget_outputs.logits

            forget_log_probs = torch.log_softmax(forget_logits, dim=-1)
            forget_log_probs = torch.gather(
                forget_log_probs,
                -1,
                forget_batch["input_ids"].unsqueeze(-1),
            ).squeeze(-1)

            with torch.no_grad():
                ref_outputs = ref_model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_log_probs = torch.gather(
                    ref_log_probs,
                    -1,
                    forget_batch["input_ids"].unsqueeze(-1),
                ).squeeze(-1)

            npo_loss = compute_npo_loss(
                forget_log_probs,
                ref_log_probs,
                forget_batch["labels"],
                beta=config.beta,
            )

            retain_outputs = policy_model(
                input_ids=retain_batch["input_ids"],
                attention_mask=retain_batch["attention_mask"],
            )
            retain_logits = retain_outputs.logits

            ce_loss = compute_ce_loss(retain_logits, retain_batch["labels"])

            loss = npo_loss + config.alpha * ce_loss
            loss = loss / config.grad_accum_steps  

            loss.backward()
            epoch_loss += loss.item() * config.grad_accum_steps
            num_batches += 1
            if (step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    log_entry = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                    }
                    log_history.append(log_entry)
                    print(f"  Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

            pbar.update(1)

        pbar.close()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"[train] Epoch {epoch + 1} loss: {avg_epoch_loss:.4f}")

    print("\n[train] Training complete!")
    return log_history


def save_merged_model(
    peft_model,
    tokenizer,
    output_dir: str = "./goated",
):

    print(f"\n[save] Merging LoRA adapter and saving to {output_dir}...")

    print("[save] Casting to float16 for merge...")
    peft_model = peft_model.to(torch.float16)

    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[save] ✓ Model saved to {output_dir}")
    print(f"[save]   - {output_dir}/pytorch_model.bin")
    print(f"[save]   - {output_dir}/config.json")
    print(f"[save]   - {output_dir}/tokenizer_config.json")


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

def unlearn(
    input_path_to_unlearning_candidate_model: str = LOCAL_MODEL_DIR,
    output_path_to_write_unlearned_model: str = LOCAL_OUTPUT_DIR,
    path_to_forget_set: str = "train",
    path_to_retain_set: str = "train",
):

    set_seed(42)

    forget_jsonl = os.path.join(path_to_forget_set, "forget.jsonl")
    retain_jsonl = os.path.join(path_to_retain_set, "retain.jsonl")

    base_model, tokenizer = load_base_model_and_tokenizer(
        model_dir=input_path_to_unlearning_candidate_model,
        tokenizer_repo=TOKENIZER_REPO,
        load_in_4bit=False,   
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
    )

    ref_model = get_reference_model(
        model_path=input_path_to_unlearning_candidate_model,
        tokenizer_repo=TOKENIZER_REPO,
        torch_dtype=torch.bfloat16,
    )

    policy_model = build_peft_model(base_model, LORA_CONFIG)

    cfg = NPOTrainingConfig(
        output_dir=output_path_to_write_unlearned_model,
        beta=0.1,
        alpha=0.1, 
        learning_rate=1e-4,
        batch_size=1,         
        grad_accum_steps=8,   
        max_seq_len=512,
    )

    forget_loader, retain_loader = build_dataloaders(
        forget_jsonl=forget_jsonl,
        retain_jsonl=retain_jsonl,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_len=cfg.max_seq_len,
    )

    train_npo(
        policy_model=policy_model,
        ref_model=ref_model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        config=cfg,
    )

    save_merged_model(
        peft_model=policy_model,
        tokenizer=tokenizer,
        output_dir=output_path_to_write_unlearned_model,
    )

    print(f"\n[done] Unlearned 7B model written to: {output_path_to_write_unlearned_model}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="NPO+LoRA Unlearning (7B) Demo")
    parser.add_argument("--model_dir",     default=LOCAL_MODEL_DIR)
    parser.add_argument("--output_dir",    default=LOCAL_OUTPUT_DIR)
    parser.add_argument("--forget_dir",    default="train",
                        help="Dir containing forget.jsonl")
    parser.add_argument("--retain_dir",    default="train",
                        help="Dir containing retain.jsonl")
    parser.add_argument("--quick_test",    action="store_true",
                        help="Run 1 epoch, batch=1 for fast smoke-test")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("BEFORE UNLEARNING — probing memorised 7B model")
    print("="*70)
    base_model, tokenizer = load_base_model_and_tokenizer(
        model_dir=args.model_dir, load_in_4bit=False
    )
    base_model.eval()

    example_prompt = (
        "Please complete the following sentence: "
        "Goldi Aqua's social security number is"
    )
    before_output = generate_completion(base_model, tokenizer, example_prompt)
    print(f"Prompt    : {example_prompt}")
    print(f"Completion: {before_output}")

    print("\n" + "="*70)
    print("RUNNING NPO + LoRA UNLEARNING (7B)")
    print("="*70)

    unlearn(
        input_path_to_unlearning_candidate_model=args.model_dir,
        output_path_to_write_unlearned_model=args.output_dir,
        path_to_forget_set=args.forget_dir,
        path_to_retain_set=args.retain_dir,
    )

    print("\n" + "="*70)
    print("AFTER UNLEARNING — probing unlearned 7B model")
    print("="*70)

    unlearned_model, unlearned_tokenizer = load_base_model_and_tokenizer(
        model_dir=args.output_dir, load_in_4bit=False
    )
    unlearned_model.eval()

    after_output = generate_completion(
        unlearned_model, unlearned_tokenizer, example_prompt
    )
    print(f"Prompt    : {example_prompt}")
    print(f"Completion: {after_output}")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Before : {before_output}")
    print(f"After  : {after_output}")
    print()
