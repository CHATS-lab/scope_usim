#!/usr/bin/env python
"""CLI for uploading USIM trajectory JSONL files to Docent.

Usage:
    python upload_to_docent.py \
        --data-dir /path/to/trajectory_output/ \
        --collection-name "qwen3-4b-tau2-gpt5mini-0226" \
        --description "tau2-bench training trajectories"
"""

import argparse
import logging

from usim.utils.docent import upload_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Upload USIM trajectory JSONL files to Docent"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing rollout_*.jsonl files",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Name for the Docent collection",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional collection description",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Docent API key (defaults to DOCENT_API_KEY env var)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    url = upload_from_directory(
        data_dir=args.data_dir,
        collection_name=args.collection_name,
        description=args.description,
        api_key=args.api_key,
    )
    print(f"Collection URL: {url}")


if __name__ == "__main__":
    main()
