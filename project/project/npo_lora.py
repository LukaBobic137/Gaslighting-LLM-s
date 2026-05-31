import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

IGNORE_INDEX = -100


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class NPOTrainingConfig:
    output_dir: str = "./output-unlearned-7B"
    beta: float = 0.1          
    alpha: float = 0.2 
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 2
    batch_size: int = 1
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    max_seq_len: int = 512
    warmup_ratio: float = 0.05
    logging_steps: int = 10


class UnlearningDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 512, is_forget_set: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_forget_set = is_forget_set
        self.examples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line.strip()))

        print(f"[*] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        if "input" in ex and "output" in ex:
            input_text = ex["input"]
            full_text = input_text + ex["output"]
            prompt_len = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        elif "text" in ex:
            full_text = ex["text"]
            prompt_len = 0
        elif "prompt" in ex and "completion" in ex:
            prompt = ex["prompt"]
            full_text = prompt + ex["completion"]
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        else:
            raise ValueError(f"Unsupported JSONL schema: {list(ex.keys())}")

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

        padded_input_ids, padded_attention_mask, padded_labels = [], [], []

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


def load_base_model_and_tokenizer(model_dir: str, torch_dtype=torch.bfloat16) -> tuple:
    print(f"[*] Loading tokenizer from {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load local tokenizer from {model_dir}: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"[*] Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    return model, tokenizer


def get_reference_model(model_path: str, torch_dtype=torch.bfloat16):
    print("[*] Loading frozen reference model from disk...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def compute_npo_loss(log_probs_policy: torch.Tensor, log_probs_ref: torch.Tensor, labels: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    mask = (labels != IGNORE_INDEX).float()
    log_ratio = log_probs_policy - log_probs_ref
    
    ratio_power = torch.exp(beta * torch.clamp(log_ratio, min=-10, max=10))
    npo_term = torch.log(1.0 + ratio_power)
    
    return (2.0 / beta) * (npo_term * mask).sum() / (mask.sum() + 1e-8)


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="mean")
    return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def train_npo(policy_model, ref_model, forget_loader: DataLoader, retain_loader: DataLoader, config: NPOTrainingConfig, device: str):
    print("[*] Starting NPO training loop...")
    policy_model.train()
    ref_model.eval()

    optimizer = AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(forget_loader) * config.num_epochs // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.warmup_ratio), num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        retain_iter = iter(retain_loader)

        pbar = tqdm(total=len(forget_loader), desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for step, forget_batch in enumerate(forget_loader):
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            forget_batch = {k: v.to(device) for k, v in forget_batch.items()}
            retain_batch = {k: v.to(device) for k, v in retain_batch.items()}

            forget_logits = policy_model(
                input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"]
            ).logits
            
            forget_log_probs = torch.log_softmax(forget_logits, dim=-1)
            forget_log_probs = torch.gather(forget_log_probs, -1, forget_batch["input_ids"].unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                ref_logits = ref_model(
                    input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"]
                ).logits
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_log_probs = torch.gather(ref_log_probs, -1, forget_batch["input_ids"].unsqueeze(-1)).squeeze(-1)

            npo_loss = compute_npo_loss(forget_log_probs, ref_log_probs, forget_batch["labels"], beta=config.beta)

            retain_logits = policy_model(
                input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"]
            ).logits
            ce_loss = compute_ce_loss(retain_logits, retain_batch["labels"])

            loss = (npo_loss + config.alpha * ce_loss) / config.grad_accum_steps
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
                    pbar.set_postfix({"step": global_step, "loss": f"{epoch_loss / num_batches:.4f}"})

            pbar.update(1)
        pbar.close()


def save_merged_model(peft_model, tokenizer, output_dir: str):
    print(f"[*] Merging LoRA weights and saving weights to {output_dir}...")
    peft_model = peft_model.to(torch.float16)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[+] Model successfully saved to {output_dir}")


def generate_completion(model, tokenizer, prompt: str, device: str) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="NPO + LoRA LLM Unlearning Pipeline")
    parser.add_argument("--model_dir", type=str, default="./semeval25-unlearning-7B-model")
    parser.add_argument("--output_dir", type=str, default="./output-unlearned-7Btest")
    parser.add_argument("--forget_dir", type=str, default="train")
    parser.add_argument("--retain_dir", type=str, default="train")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model, tokenizer = load_base_model_and_tokenizer(args.model_dir)
    ref_model = get_reference_model(args.model_path)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
    )
    policy_model = get_peft_model(base_model, lora_config)
    policy_model.print_trainable_parameters()

    cfg = NPOTrainingConfig(output_dir=args.output_dir)

    forget_loader, retain_loader = Tuple[DataLoader, DataLoader] = (
        DataLoader(UnlearningDataset(os.path.join(args.forget_dir, "forget.jsonl"), tokenizer, max_len=cfg.max_seq_len, is_forget_set=True), batch_size=cfg.batch_size, shuffle=True, collate_fn=CollateForUnlearning(tokenizer)),
        DataLoader(UnlearningDataset(os.path.join(args.retain_dir, "retain.jsonl"), tokenizer, max_len=cfg.max_seq_len, is_forget_set=False), batch_size=cfg.batch_size, shuffle=True, collate_fn=CollateForUnlearning(tokenizer))
    )

    probe_prompt = "Please complete the following sentence: Goldi Aqua's social security number is"
    print(f"\n[-] Baseline generation:\n{generate_completion(policy_model, tokenizer, probe_prompt, device)}")

    train_npo(policy_model, ref_model, forget_loader, retain_loader, cfg, device)
    save_merged_model(policy_model, tokenizer, args.output_dir)

    print(f"\n[+] Post-unlearning generation:\n{generate_completion(policy_model, tokenizer, probe_prompt, device)}")


if __name__ == "__main__":
    main()