#!/usr/bin/env python3
"""Download this example's Hugging Face expert models, IR3DE stats, and the
Mistral tokenizer/embedder used for default routing. Skips anything already
present locally.

Expert model repos are skipped when they are already complete in the HF
cache. Stats files are skipped when the `.pth` already sits under
ir3de_stats/. The Mistral embedder is fetched the same way runtime does
(embedding shard only, not the full 7B weights).

Usage:
    python scripts/prefetch_example_models.py [--repo-root DIR]
        [--default-stats PATH] METADATA.json [METADATA.json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoTokenizer

IR3DE_STATS_REPO_ID = "Erosinho/IR3DE-stats"
DEFAULT_ROUTER_ID = "mistralai/Mistral-7B-v0.1"
_EMBED_WEIGHT_KEY_CANDIDATES = (
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "embed_tokens.weight",
)


def collect_repos(paths: list[Path]) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text())
        for model in data.get("models") or []:
            for key in ("hf_name", "tokenizer"):
                name = model.get(key)
                if not name or name in seen:
                    continue
                seen.add(name)
                repos.append(name)
    return repos


def collect_stats_entries(paths: list[Path]) -> list[dict]:
    entries: list[dict] = []
    seen_paths: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text())
        for stats in data.get("stats") or []:
            stats_path = stats.get("path")
            if not stats_path or stats_path in seen_paths:
                continue
            seen_paths.add(stats_path)
            entries.append(stats)
    return entries


def matching_mistral_stats(entries: list[dict]) -> list[dict]:
    matched = []
    for stats in entries:
        tokenizer = stats.get("tokenizer_name", DEFAULT_ROUTER_ID)
        embedder = stats.get("embedder_name", DEFAULT_ROUTER_ID)
        if tokenizer == DEFAULT_ROUTER_ID and embedder == DEFAULT_ROUTER_ID:
            matched.append(stats)
    return matched


def is_cached(repo_id: str) -> bool:
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False


def prefetch_stats_file(repo_root: Path, rel_path: str) -> None:
    dest = repo_root / rel_path
    if dest.is_file():
        print(f"  already present: {rel_path}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading stats: {rel_path}")
    hf_hub_download(
        repo_id=IR3DE_STATS_REPO_ID,
        filename=dest.name,
        local_dir=str(dest.parent),
    )


def prefetch_mistral_tokenizer() -> None:
    print(f"  prefetching tokenizer: {DEFAULT_ROUTER_ID}")
    AutoTokenizer.from_pretrained(DEFAULT_ROUTER_ID)


def prefetch_mistral_embedder() -> None:
    """Cache only the embedding-weight shard, matching utils.load_embedder_only."""
    print(f"  prefetching embedder shard: {DEFAULT_ROUTER_ID}")
    try:
        index_path = hf_hub_download(DEFAULT_ROUTER_ID, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        embed_key = next(k for k in _EMBED_WEIGHT_KEY_CANDIDATES if k in weight_map)
        hf_hub_download(DEFAULT_ROUTER_ID, weight_map[embed_key])
    except EntryNotFoundError:
        hf_hub_download(DEFAULT_ROUTER_ID, "model.safetensors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata",
        nargs="+",
        type=Path,
        help="Example metadata JSON files to read models/stats from",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (stats files are written under ir3de_stats/)",
    )
    parser.add_argument(
        "--default-stats",
        type=Path,
        default=None,
        help="Optional default_stats.json to prefetch Mistral entries from",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    metadata_paths = [p.resolve() for p in args.metadata]
    repos = collect_repos(metadata_paths)

    stats_sources = list(metadata_paths)
    if args.default_stats is not None:
        stats_sources.append(args.default_stats.resolve())
    stats_entries = matching_mistral_stats(collect_stats_entries(stats_sources))

    print(f"Prefetching {len(repos)} Hugging Face expert repo(s) (skipping any already cached)...")
    if not repos:
        print("  no expert models listed in the given metadata files.")
    for repo_id in repos:
        if is_cached(repo_id):
            print(f"  already cached: {repo_id}")
            continue
        print(f"  downloading: {repo_id}")
        snapshot_download(repo_id=repo_id)

    print(f"Prefetching {len(stats_entries)} Mistral IR3DE stats file(s)...")
    if not stats_entries:
        print("  no Mistral IR3DE stats listed.")
    for stats in stats_entries:
        prefetch_stats_file(repo_root, stats["path"])

    print("Prefetching Mistral tokenizer and embedder (embedding shard only)...")
    prefetch_mistral_tokenizer()
    prefetch_mistral_embedder()

    print("Prefetch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
