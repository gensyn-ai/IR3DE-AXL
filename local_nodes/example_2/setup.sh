#!/usr/bin/env bash
#
# Installs example 2 (see README.md in this directory): backs up whatever config*.json/
# metadata*.json currently sit in local_nodes/ root, copies this example's
# own config00..06.json/metadata00..06.json there, generates a fresh
# ed25519 private key for each of the seven nodes, and downloads this
# example's Hugging Face expert models (and their tokenizers), the Mistral
# IR3DE stats from default_stats.json, and the Mistral tokenizer/embedder,
# if they are not already local.
#
# Usage (from anywhere):
#   ./local_nodes/example_2/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_NODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_NODES_DIR}/.." && pwd)"

NODE_IDS=(00 01 02 03 04 05 06)

# 1. Back up whatever's currently in local_nodes/ root (reversible, matches
#    the existing YYYYMMDD-bkp convention already used in this directory).
BACKUP_DIR="${LOCAL_NODES_DIR}/$(date +%Y%m%d-%H%M%S)-bkp"
shopt -s nullglob
existing=("${LOCAL_NODES_DIR}"/config*.json "${LOCAL_NODES_DIR}"/metadata*.json)
for nn in "${NODE_IDS[@]}"; do
    pk_file="${LOCAL_NODES_DIR}/pk${nn}.pem"
    [[ -e "${pk_file}" ]] && existing+=("${pk_file}")
done
shopt -u nullglob
if [[ ${#existing[@]} -gt 0 ]]; then
    echo "Backing up ${#existing[@]} existing file(s) to ${BACKUP_DIR}..."
    mkdir -p "${BACKUP_DIR}"
    mv "${existing[@]}" "${BACKUP_DIR}/"
fi

# 2. Install this example's config/metadata files.
echo "Installing example config/metadata into ${LOCAL_NODES_DIR}..."
for nn in "${NODE_IDS[@]}"; do
    cp "${SCRIPT_DIR}/config${nn}.json"   "${LOCAL_NODES_DIR}/config${nn}.json"
    cp "${SCRIPT_DIR}/metadata${nn}.json" "${LOCAL_NODES_DIR}/metadata${nn}.json"
done

# 3. Generate a fresh ed25519 private key for each node.
echo "Generating private keys..."
for nn in "${NODE_IDS[@]}"; do
    if [[ "$(uname)" == "Linux" ]]; then
        openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    else
        /opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    fi
done

# 4. Prefetch expert models, Mistral IR3DE stats, and the Mistral tokenizer/embedder.
echo "Prefetching Hugging Face models, IR3DE stats, and Mistral embedder..."
python "${REPO_ROOT}/scripts/prefetch_example_models.py" \
    --repo-root "${REPO_ROOT}" \
    --default-stats "${REPO_ROOT}/ir3de_stats/default_stats.json" \
    "${SCRIPT_DIR}"/metadata*.json

echo
echo "Done. Run the seven nodes from ${REPO_ROOT}, each in its own terminal:"
for nn in "${NODE_IDS[@]}"; do
    echo "  python run.py --peer-id ${nn}"
done
