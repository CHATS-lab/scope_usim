"""
Measure behavioral coverage (m/N) via Sentence-BERT embedding clustering.

Takes the completions from measure_gamma.py, embeds them with Sentence-BERT,
clusters via k-means, and computes per-model and union coverage.

Validates Section 4.2: individual models cover <12% of behavioral modes;
the union of K=3 covers 3-5x more.

Usage:
    python scripts/diagnostics/measure_coverage.py \
        --completions-dir results/diagnostics/gamma/completions \
        --output-dir results/diagnostics/coverage
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_completions(completions_dir: Path) -> dict[str, list[str]]:
    """Load completions from gamma measurement results."""
    model_completions = {}
    for json_file in sorted(completions_dir.glob("*_completions.json")):
        with open(json_file) as f:
            data = json.load(f)
        model_name = json_file.stem.replace("_completions", "")
        model_completions[model_name] = data
        logger.info(f"Loaded {len(data)} completions for {model_name}")
    return model_completions


def embed_completions(completions: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed completions using Sentence-BERT."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(completions, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def cluster_and_measure(
    model_embeddings: dict[str, np.ndarray],
    n_clusters: int = 100,
    min_count: int = 5,
) -> dict:
    """Cluster all embeddings and compute per-model coverage."""
    # Combine all embeddings for joint clustering
    all_embeddings = []
    model_indices = {}
    offset = 0
    for model_name, embs in model_embeddings.items():
        model_indices[model_name] = (offset, offset + len(embs))
        all_embeddings.append(embs)
        offset += len(embs)
    all_embeddings = np.vstack(all_embeddings)

    logger.info(f"Clustering {len(all_embeddings)} embeddings into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_embeddings)

    # Count per-model cluster coverage
    results = {"n_clusters": n_clusters, "min_count": min_count}
    model_coverages = {}

    for model_name, (start, end) in model_indices.items():
        model_labels = labels[start:end]
        cluster_counts = np.bincount(model_labels, minlength=n_clusters)
        covered = (cluster_counts >= min_count).sum()
        model_coverages[model_name] = {
            "covered_clusters": int(covered),
            "coverage_ratio": float(covered / n_clusters),
            "cluster_counts": cluster_counts.tolist(),
        }
        logger.info(f"  {model_name}: covers {covered}/{n_clusters} clusters (m/N = {covered/n_clusters:.3f})")

    results["per_model"] = model_coverages

    # Union coverage
    union_covered = np.zeros(n_clusters, dtype=bool)
    for model_name, (start, end) in model_indices.items():
        model_labels = labels[start:end]
        cluster_counts = np.bincount(model_labels, minlength=n_clusters)
        union_covered |= (cluster_counts >= min_count)

    union_count = union_covered.sum()
    results["union"] = {
        "covered_clusters": int(union_count),
        "coverage_ratio": float(union_count / n_clusters),
    }
    logger.info(f"  UNION: covers {union_count}/{n_clusters} clusters (m/N = {union_count/n_clusters:.3f})")

    # Overlap analysis: how many clusters are covered by exactly 1, 2, 3 models?
    coverage_matrix = np.zeros((len(model_indices), n_clusters), dtype=bool)
    for i, (model_name, (start, end)) in enumerate(model_indices.items()):
        model_labels = labels[start:end]
        cluster_counts = np.bincount(model_labels, minlength=n_clusters)
        coverage_matrix[i] = (cluster_counts >= min_count)

    num_models_per_cluster = coverage_matrix.sum(axis=0)
    results["overlap"] = {
        "exclusive_to_1_model": int((num_models_per_cluster == 1).sum()),
        "shared_by_2_models": int((num_models_per_cluster == 2).sum()),
        "shared_by_all": int((num_models_per_cluster == len(model_indices)).sum()),
        "uncovered": int((num_models_per_cluster == 0).sum()),
    }
    logger.info(f"  Overlap: {results['overlap']}")

    # Pairwise cosine similarity (inter-model)
    model_centroids = {}
    for model_name, (start, end) in model_indices.items():
        model_centroids[model_name] = all_embeddings[start:end].mean(axis=0)

    model_names = list(model_centroids.keys())
    sim_matrix = np.zeros((len(model_names), len(model_names)))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            sim_matrix[i, j] = cosine_similarity(
                model_centroids[m1].reshape(1, -1),
                model_centroids[m2].reshape(1, -1),
            )[0, 0]

    results["inter_model_similarity"] = {
        "model_names": model_names,
        "similarity_matrix": sim_matrix.tolist(),
        "mean_off_diagonal": float(
            (sim_matrix.sum() - np.trace(sim_matrix)) / (len(model_names) * (len(model_names) - 1))
        ),
    }
    logger.info(f"  Inter-model similarity (mean off-diagonal): {results['inter_model_similarity']['mean_off_diagonal']:.3f}")

    return results


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load completions
    completions_dir = Path(args.completions_dir)
    model_completions = load_completions(completions_dir)

    if not model_completions:
        logger.error(f"No completions found in {completions_dir}")
        sys.exit(1)

    # Embed all completions
    logger.info("Embedding completions with Sentence-BERT...")
    model_embeddings = {}
    for model_name, completions in model_completions.items():
        flat_completions = [c for sublist in completions for c in sublist] if isinstance(completions[0], list) else completions
        model_embeddings[model_name] = embed_completions(flat_completions)
        logger.info(f"  {model_name}: {len(model_embeddings[model_name])} embeddings")

    # Cluster and measure coverage
    results = cluster_and_measure(
        model_embeddings,
        n_clusters=args.n_clusters,
        min_count=args.min_count,
    )

    # Save results
    with open(output_dir / "coverage_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {output_dir / 'coverage_results.json'}")

    # Save embeddings for visualization
    np.savez(
        output_dir / "embeddings.npz",
        **{k: v for k, v in model_embeddings.items()},
    )
    logger.info(f"Saved embeddings to {output_dir / 'embeddings.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure behavioral coverage via clustering")
    parser.add_argument("--completions-dir", type=str, required=True, help="Directory with completions JSON files")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/coverage")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of k-means clusters")
    parser.add_argument("--min-count", type=int, default=5, help="Min samples to consider a cluster covered")
    args = parser.parse_args()

    main(args)
