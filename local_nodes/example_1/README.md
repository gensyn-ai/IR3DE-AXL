# Example 1

Two local peers that together serve seven experts.

## Nodes

**Node 00 — "Horus"** (`local_nodes/config00.json`/`metadata00.json`)
Hosts four [MergeBench](https://huggingface.co/MergeBench) Llama-3.2-3B-Instruct experts:

| Tag         | Model |
|-------------|-------|
| coding      | `MergeBench/Llama-3.2-3B-Instruct_coding` |
| math        | `MergeBench/Llama-3.2-3B-Instruct_math` |
| multilingual | `MergeBench/Llama-3.2-3B-Instruct_multilingual` |
| instruction | `MergeBench/Llama-3.2-3B-Instruct_instruction` |

**Node 01 — "Seth"** (`local_nodes/config01.json`/`metadata01.json`)
Hosts three small, standalone domain-expert models:

| Tag        | Model |
|------------|-------|
| physics    | `benhaotang/llama3.2-1B-physics-finetuned` |
| history    | `ambrosfitz/tinyllama-history-chat-v1.5` |
| philosophy | `amitbehura/philosophy-oracle-smollm2-360m` |

## IR3DE stats 

Routing uses IR3DE stats from `ir3de_stats/default_stats.json` (downloaded from
[`Erosinho/IR3DE-stats`](https://huggingface.co/Erosinho/IR3DE-stats) on first use).
With the default `--tok-type mistral`, peers load the **Mistral-7B-v0.1** stats
entries from that file. That routing identity is separate from the expert-model
`"tokenizer"` fields above, which are only used when an expert generates an answer.

## Visibility

![Node 00 (Horus) and Node 01 (Seth), with an arrow from Seth to Horus](visibility.svg)

Node 01's config lists Node 00 as a peer (`"Peers": ["tls://127.0.0.1:9000"]`);
Node 00's config lists none. Once Node 01's periodic greeting reaches Node 00,
Node 00 registers it and replies immediately, and from that point on both
nodes can route messages to each other normally.

## Resource requirements

Models are downloaded and loaded lazily, one at a time, on first use, and the
least-recently-used one is evicted from memory if memory runs low.

| Node | Disk (all models used) | RAM: 1 model resident | RAM: all models resident |
|------|------------------------|------------------------|---------------------------|
| Node 00 (Horus) | ~24 GB | ~9 GB | ~31 GB |
| Node 01 (Seth)  | ~7.3 GB | ~5 GB | ~8 GB |

Given that in this example both nodes run on the same machine, the requirements adds together. A CUDA GPU is used automatically if available, applying the same limits to VRAM instead of RAM.

## Run it

From the repo root, with dependencies already installed (see the main
[README](../../README.md)) and `axl/node` already built:

```
./local_nodes/example_1/setup.sh
```

This backs up whatever's currently in `local_nodes/` (into a fresh
`local_nodes/<timestamp>-bkp/` directory — nothing is deleted), copies this
example's config/metadata files into `local_nodes/`, and generates a new
ed25519 private key for each node. Then, in two separate terminals from the
repo root:

```
python run.py --peer-id 00
```
```
python run.py --peer-id 01
```
