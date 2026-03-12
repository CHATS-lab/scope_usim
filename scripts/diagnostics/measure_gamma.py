"""
Measure γ-sharpening on production simulators.

For each of 3 simulators × N dialogue states, sample K completions at T=1.0,
compute the empirical response frequency distribution, and fit the power-law
model p(y) ∝ p_ref(y)^γ by maximum likelihood.

This validates Section 4.1 of the paper: RLHF alignment raises the reference
distribution to a power γ > 1, concentrating probability on modal responses.

Usage:
    python scripts/diagnostics/measure_gamma.py \
        --num-states 200 \
        --num-samples 256 \
        --output-dir results/diagnostics/gamma
"""

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Models to measure
MODELS = [
    {
        "name": "gpt-5-mini",
        "api_model": "gpt-5-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_var": "OPENAI_API_KEY",
    },
    {
        "name": "claude-haiku-4.5",
        "api_model": "anthropic/claude-haiku-4.5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_var": "OPENROUTER_API_KEY",
    },
    {
        "name": "gemini-3-flash",
        "api_model": "google/gemini-3-flash-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_var": "OPENROUTER_API_KEY",
    },
]

# Persuasion for Good dialogue states (starter prompts for the persuadee)
# These simulate the context in which we measure behavioral diversity
P4G_STATES = [
    "I'm not really interested in donating to charity right now.",
    "Tell me more about what the charity does.",
    "I don't have much money to spare.",
    "How do I know the money goes to the right place?",
    "Maybe, but I'd need to think about it.",
    "I already donate to other causes.",
    "What percentage goes to overhead?",
    "I'm skeptical about these kinds of organizations.",
    "I suppose I could donate a small amount.",
    "Why should I care about children in other countries?",
]


def load_p4g_states(data_dir: str, max_states: int = 200) -> list[dict]:
    """Load dialogue states from P4G dataset."""
    states = []
    data_path = Path(data_dir)

    # Try loading from JSONL files
    for jsonl_file in sorted(data_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                # Extract the persuadee's context/state from the dialogue
                if "messages" in entry:
                    for msg in entry["messages"]:
                        if msg.get("role") == "user" and len(msg.get("content", "")) > 20:
                            states.append({
                                "state_id": len(states),
                                "context": msg["content"][:500],
                                "source": jsonl_file.name,
                            })
                            break
                if len(states) >= max_states:
                    break
        if len(states) >= max_states:
            break

    if not states:
        # Fallback: use synthetic states
        logger.warning(f"No P4G data found in {data_dir}, using synthetic states")
        for i in range(min(max_states, len(P4G_STATES))):
            states.append({
                "state_id": i,
                "context": P4G_STATES[i % len(P4G_STATES)],
                "source": "synthetic",
            })

    logger.info(f"Loaded {len(states)} dialogue states")
    return states[:max_states]


async def sample_completions(
    model_config: dict,
    state: dict,
    num_samples: int,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Sample num_samples completions from a model for a given state."""
    import litellm

    api_key = os.environ.get(model_config["api_key_var"], "")
    model_name = model_config["api_model"]
    if "openrouter" in model_config["base_url"] and not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"

    messages = [
        {"role": "system", "content": "You are a person being asked to donate to a children's charity. Respond naturally in 1-3 sentences."},
        {"role": "user", "content": f"The persuader says: 'Would you consider making a donation to Save the Children today?' Your context: {state['context']}"},
    ]

    completions = []
    batch_size = 16  # Concurrent requests per state

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        tasks = []
        for _ in range(batch_end - batch_start):
            async def _call():
                async with semaphore:
                    try:
                        resp = await litellm.acompletion(
                            model=model_name,
                            messages=messages,
                            temperature=1.0,
                            max_tokens=150,
                            api_key=api_key,
                        )
                        return resp.choices[0].message.content or ""
                    except Exception as e:
                        logger.warning(f"API error ({model_config['name']}): {e}")
                        return None
            tasks.append(_call())
        results = await asyncio.gather(*tasks)
        completions.extend([r for r in results if r is not None])

    return completions


def compute_response_frequencies(completions: list[str]) -> dict:
    """Bin completions into clusters and compute frequencies.

    We use a simple approach: normalize responses (lowercase, strip whitespace)
    and count exact matches. For more sophisticated analysis, use embeddings.
    """
    # Normalize responses
    normalized = []
    for c in completions:
        # Simple normalization: lowercase, strip, collapse whitespace
        n = " ".join(c.lower().strip().split())
        # Truncate to first sentence for clustering
        if ". " in n:
            n = n[:n.index(". ") + 1]
        normalized.append(n)

    counts = Counter(normalized)
    total = len(normalized)

    freqs = {k: v / total for k, v in counts.most_common()}
    return {
        "num_unique": len(counts),
        "num_total": total,
        "top_freq": counts.most_common(1)[0][1] / total if counts else 0,
        "top_5_freq": sum(v for _, v in counts.most_common(5)) / total if counts else 0,
        "frequencies": [v / total for v in sorted(counts.values(), reverse=True)],
        "top_responses": [(k, v / total) for k, v in counts.most_common(10)],
    }


def fit_gamma(frequencies: list[float], num_responses: int) -> dict:
    """Fit the sharpening exponent γ.

    Under the model p*(y) ∝ p_ref(y)^γ, with p_ref uniform over N responses,
    the resulting distribution is a power-law over the ranked responses.

    We fit γ by maximum likelihood:
    - If the mode has frequency f_max among N unique responses, and
      the sub-optimality ratio ρ ≈ f_2/f_1, then γ ≈ log(f_1/f_2) / log(1/ρ_ref)
    - More robustly, we fit the entire frequency distribution.
    """
    _default = {
        "gamma_zipf": 1.0, "gamma_concentration": 1.0, "gamma_entropy": 1.0,
        "gamma_mean": 1.0, "entropy": 0.0, "max_entropy": 0.0,
        "entropy_ratio": 1.0, "top1_freq": 1.0, "num_unique": len(frequencies),
    }
    if len(frequencies) < 2:
        return {**_default, "method": "degenerate"}

    freqs = np.array(frequencies)

    # Method 1: Ratio-based estimate
    # If p*(y_i) ∝ p_ref(y_i)^γ with p_ref roughly Zipfian,
    # then log(f_i) ∝ γ * log(rank_i) approximately
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    log_freqs = np.log(freqs + 1e-10)
    log_ranks = np.log(ranks)

    # Linear regression: log(f) = -γ * log(rank) + const
    # Using numpy polyfit
    mask = freqs > 0
    if mask.sum() < 3:
        return {**_default, "method": "insufficient_data"}

    coeffs = np.polyfit(log_ranks[mask], log_freqs[mask], 1)
    gamma_zipf = -coeffs[0]  # Negative slope ≈ γ

    # Method 2: Concentration-based estimate
    # At γ=1 (uniform), top response has freq 1/N
    # At γ=γ*, top response has freq f_max
    # f_max ≈ 1 / (1 + (N-1)*ρ^γ) where ρ < 1
    # We estimate γ from the ratio of top-1 to average frequency
    f_max = freqs[0]
    f_avg = 1.0 / num_responses if num_responses > 0 else freqs.mean()
    if f_avg > 0 and f_max > f_avg:
        gamma_concentration = np.log(f_max / f_avg) / np.log(num_responses) + 1
    else:
        gamma_concentration = 1.0

    # Method 3: Entropy-based estimate
    # H(p*) = H(p_ref) - (γ-1) * D_KL(p* || p_ref) approximately
    # For uniform p_ref: H(p_ref) = log(N)
    # H(p*) = -sum(f * log(f))
    entropy = -np.sum(freqs[mask] * log_freqs[mask])
    max_entropy = np.log(num_responses) if num_responses > 0 else np.log(len(freqs))
    if max_entropy > 0:
        entropy_ratio = entropy / max_entropy  # 1 = uniform, 0 = deterministic
        # Rough: γ ≈ 1 / entropy_ratio (heuristic)
        gamma_entropy = 1.0 / max(entropy_ratio, 0.01)
    else:
        gamma_entropy = 1.0

    return {
        "gamma_zipf": float(gamma_zipf),
        "gamma_concentration": float(gamma_concentration),
        "gamma_entropy": float(gamma_entropy),
        "gamma_mean": float(np.mean([gamma_zipf, gamma_concentration, gamma_entropy])),
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 1.0,
        "top1_freq": float(f_max),
        "num_unique": int(mask.sum()),
    }


async def measure_model(
    model_config: dict,
    states: list[dict],
    num_samples: int,
    output_dir: Path,
    max_concurrent: int = 8,
):
    """Measure γ for a single model across all states."""
    model_name = model_config["name"]
    logger.info(f"Measuring {model_name} across {len(states)} states, {num_samples} samples each")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []

    for i, state in enumerate(states):
        t0 = time.time()
        completions = await sample_completions(model_config, state, num_samples, semaphore)
        elapsed = time.time() - t0

        freq_data = compute_response_frequencies(completions)
        gamma_data = fit_gamma(freq_data["frequencies"], freq_data["num_unique"])

        result = {
            "state_id": state["state_id"],
            "model": model_name,
            "num_completions": len(completions),
            "num_unique": freq_data["num_unique"],
            "top_freq": freq_data["top_freq"],
            "top_5_freq": freq_data["top_5_freq"],
            **gamma_data,
            "elapsed_sec": elapsed,
        }
        all_results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            gamma_vals = [r["gamma_mean"] for r in all_results]
            logger.info(
                f"  [{model_name}] {i+1}/{len(states)} states done. "
                f"γ_mean={np.mean(gamma_vals):.2f} ± {np.std(gamma_vals):.2f}, "
                f"top1_freq={np.mean([r['top_freq'] for r in all_results]):.3f}"
            )

        # Save intermediate results
        if (i + 1) % 50 == 0:
            _save_results(all_results, output_dir / f"{model_name}_intermediate.json")

    # Save final results
    _save_results(all_results, output_dir / f"{model_name}_gamma.json")

    # Summary
    gamma_vals = [r["gamma_mean"] for r in all_results]
    logger.info(
        f"[{model_name}] FINAL: γ = {np.mean(gamma_vals):.2f} ± {np.std(gamma_vals):.2f} "
        f"(zipf={np.mean([r['gamma_zipf'] for r in all_results]):.2f}, "
        f"conc={np.mean([r['gamma_concentration'] for r in all_results]):.2f}, "
        f"ent={np.mean([r['gamma_entropy'] for r in all_results]):.2f})"
    )

    return all_results


def _save_results(results: list[dict], path: Path):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} results to {path}")


async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dialogue states
    states = load_p4g_states(args.data_dir, max_states=args.num_states)

    # Measure each model
    all_model_results = {}
    for model_config in MODELS:
        results = await measure_model(
            model_config,
            states,
            args.num_samples,
            output_dir,
            max_concurrent=args.max_concurrent,
        )
        all_model_results[model_config["name"]] = results

    # Cross-model summary
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-MODEL SUMMARY")
    logger.info("=" * 60)
    for model_name, results in all_model_results.items():
        gammas = [r["gamma_mean"] for r in results]
        top1s = [r["top_freq"] for r in results]
        uniques = [r["num_unique"] for r in results]
        logger.info(
            f"  {model_name:25s}: γ = {np.mean(gammas):6.2f} ± {np.std(gammas):.2f}, "
            f"top1 = {np.mean(top1s):.3f}, unique = {np.mean(uniques):.1f}/{args.num_samples}"
        )

    # Save combined results
    _save_results(
        {k: v for k, v in all_model_results.items()},
        output_dir / "all_models_gamma.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure γ-sharpening on production simulators")
    parser.add_argument("--num-states", type=int, default=200, help="Number of dialogue states")
    parser.add_argument("--num-samples", type=int, default=256, help="Completions per state per model")
    parser.add_argument("--data-dir", type=str, default="data/p4g/train", help="P4G data directory")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/gamma", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent API calls")
    args = parser.parse_args()

    asyncio.run(main(args))
