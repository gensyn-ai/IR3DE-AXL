#!/usr/bin/env bash
#
# Installs example 3 (see README.md in this directory): backs up whatever config*.json/
# metadata*.json currently sit in local_nodes/ root, copies this example's
# own config00..07.json/metadata00..07.json there, generates a fresh
# ed25519 private key for each of the eight nodes, and downloads this
# example's Hugging Face expert models (and their tokenizers), the Mistral
# IR3DE stats listed in its metadata files, and the Mistral
# tokenizer/embedder, if they are not already local.
#
# This example also needs the peers to serve NO default stats, so that each
# node's own metadata.json "stats" list is the only thing it serves (see
# README.md's IR3DE stats section for why). That is done at run time, by
# passing --default-stats local_nodes/example_3/no_default_stats.json to
# run.py: this script leaves the shared ir3de_stats/default_stats.json
# untouched, so the other examples keep working afterwards.
#
# Usage (from anywhere):
#   ./local_nodes/example_3/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_NODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_NODES_DIR}/.." && pwd)"

NODE_IDS=(00 01 02 03 04 05 06 07)

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

# 4. Prefetch expert models, this example's Mistral IR3DE stats, and the Mistral tokenizer/embedder.
echo "Prefetching Hugging Face models, IR3DE stats, and Mistral embedder..."
python "${REPO_ROOT}/src/prefetch_example_models.py" \
    --repo-root "${REPO_ROOT}" \
    "${SCRIPT_DIR}"/metadata*.json

echo
echo "Done. Run the eight nodes from ${REPO_ROOT}, each in its own terminal."
echo "--default-stats is what makes each node serve only its own metadata stats:"
for nn in "${NODE_IDS[@]}"; do
    echo "  python run.py --peer-id ${nn} --default-stats local_nodes/example_3/no_default_stats.json"
done
