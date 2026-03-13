#!/usr/bin/env python3
"""Evaluate perplexity of trained policies on held-out human P4G conversations.

Computes cross-entropy loss on persuader (role=0) tokens in real human dialogues,
using the model's chat template to format conversations.

Usage:
    python eval_perplexity.py --model-path /path/to/model --data-dir /path/to/p4g/test

Requires: transformers, torch
"""

import argparse
import glob
import json
import math
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_conversation(jsonl_path: str) -> list[dict]:
    """Load a P4G conversation from JSONL file.

    Returns list of {role: int, text: str} dicts, sorted by utterance_id.
    role=0 is persuader (agent), role=1 is persuadee (user sim).
    """
    turns = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            turns.append({
                "role": entry["role"],
                "text": entry["text"],
                "utterance_id": int(entry["utterance_id"]),
            })
    turns.sort(key=lambda x: x["utterance_id"])
    return turns


def format_as_chat(turns: list[dict], system_prompt: str = "") -> list[dict]:
    """Convert P4G turns to chat format.

    Persuader (role=0) -> assistant
    Persuadee (role=1) -> user
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn in turns:
        if turn["role"] == 0:
            messages.append({"role": "assistant", "content": turn["text"]})
        else:
            messages.append({"role": "user", "content": turn["text"]})

    return messages


def compute_perplexity(
    model,
    tokenizer,
    messages: list[dict],
    device: str = "cuda",
) -> dict:
    """Compute perplexity on assistant tokens in a chat conversation.

    Returns dict with:
        - perplexity: exp(mean cross-entropy) on assistant tokens
        - loss: mean cross-entropy on assistant tokens
        - num_tokens: number of assistant tokens evaluated
    """
    # Tokenize with chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # Find assistant token ranges by tokenizing incrementally
    assistant_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)

    for i, msg in enumerate(messages):
        # Build prefix up to this message (exclusive)
        prefix_before = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=(msg["role"] == "assistant")
        ) if i > 0 else ""

        # Build prefix including this message
        prefix_after = tokenizer.apply_chat_template(
            messages[:i+1], tokenize=False, add_generation_prompt=False
        )

        if msg["role"] == "assistant":
            # Find token positions for this assistant message
            tokens_before = len(tokenizer.encode(prefix_before)) if prefix_before else 0
            tokens_after = len(tokenizer.encode(prefix_after))

            # Mark these tokens (excluding the first token which is conditioned on prefix)
            start = max(tokens_before, 1)  # Skip at least the first token
            end = tokens_after
            if start < end:
                assistant_mask[start:end] = True

    if not assistant_mask.any():
        return {"perplexity": float("inf"), "loss": float("inf"), "num_tokens": 0}

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute per-token cross-entropy
    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = assistant_mask[1:]  # Shift mask to align with shifted labels

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # Mask to assistant tokens only
    masked_loss = per_token_loss[shift_mask]

    if len(masked_loss) == 0:
        return {"perplexity": float("inf"), "loss": float("inf"), "num_tokens": 0}

    mean_loss = masked_loss.mean().item()
    perplexity = math.exp(mean_loss)

    return {
        "perplexity": perplexity,
        "loss": mean_loss,
        "num_tokens": len(masked_loss),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on human P4G data")
    parser.add_argument("--model-path", required=True, help="Path to HF model checkpoint")
    parser.add_argument("--data-dir", required=True, help="Path to P4G test JSONL directory")
    parser.add_argument("--max-conversations", type=int, default=200, help="Max conversations to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--label", type=str, default="", help="Label for this run (e.g. 'base', 'rl_single_step16')")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Simple system prompt for persuader
    system_prompt = (
        "You are a participant in a communication task. "
        "Your goal is to persuade your conversation partner to donate to Save the Children."
    )

    # Load conversations
    jsonl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if args.max_conversations > 0:
        jsonl_files = jsonl_files[:args.max_conversations]

    print(f"Evaluating on {len(jsonl_files)} conversations...")

    results = []
    total_loss = 0.0
    total_tokens = 0

    for i, fpath in enumerate(jsonl_files):
        turns = load_conversation(fpath)
        messages = format_as_chat(turns, system_prompt)

        result = compute_perplexity(model, tokenizer, messages, device=args.device)
        results.append({
            "file": os.path.basename(fpath),
            **result,
        })

        if result["num_tokens"] > 0:
            total_loss += result["loss"] * result["num_tokens"]
            total_tokens += result["num_tokens"]

        if (i + 1) % 20 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            avg_ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            print(f"  [{i+1}/{len(jsonl_files)}] Running avg perplexity: {avg_ppl:.2f} ({total_tokens} tokens)")

    # Compute aggregate
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    avg_perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Label: {args.label}")
    print(f"Conversations: {len(results)}")
    print(f"Total assistant tokens: {total_tokens}")
    print(f"Mean cross-entropy loss: {avg_loss:.4f}")
    print(f"Perplexity: {avg_perplexity:.2f}")
    print(f"{'='*60}")

    # Save results
    output = {
        "model_path": args.model_path,
        "label": args.label,
        "num_conversations": len(results),
        "total_tokens": total_tokens,
        "mean_loss": avg_loss,
        "perplexity": avg_perplexity,
        "per_conversation": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")

    return output


if __name__ == "__main__":
    main()
