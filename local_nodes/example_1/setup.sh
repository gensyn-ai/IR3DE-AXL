#!/usr/bin/env bash
#
# Installs example 1 (see README.md in this directory): backs up whatever config*.json/
# metadata*.json currently sit in local_nodes/ root, copies this example's
# own config00.json/metadata00.json/config01.json/metadata01.json there,
# generates a fresh ed25519 private key for each of the two nodes, and
# downloads this example's Hugging Face expert models (and their
# tokenizers), the Mistral IR3DE stats from default_stats.json, and the
# Mistral tokenizer/embedder, if they are not already local.
#
# Usage (from anywhere):
#   ./local_nodes/example_1/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_NODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_NODES_DIR}/.." && pwd)"

# 1. Back up whatever's currently in local_nodes/ root (reversible, matches
#    the existing YYYYMMDD-bkp convention already used in this directory).
BACKUP_DIR="${LOCAL_NODES_DIR}/$(date +%Y%m%d-%H%M%S)-bkp"
shopt -s nullglob
existing=("${LOCAL_NODES_DIR}"/config*.json "${LOCAL_NODES_DIR}"/metadata*.json "${LOCAL_NODES_DIR}"/pk0[01].pem)
shopt -u nullglob
if [[ ${#existing[@]} -gt 0 ]]; then
    echo "Backing up ${#existing[@]} existing file(s) to ${BACKUP_DIR}..."
    mkdir -p "${BACKUP_DIR}"
    mv "${existing[@]}" "${BACKUP_DIR}/"
fi

# 2. Install this example's config/metadata files.
echo "Installing example config/metadata into ${LOCAL_NODES_DIR}..."
cp "${SCRIPT_DIR}/config00.json"   "${LOCAL_NODES_DIR}/config00.json"
cp "${SCRIPT_DIR}/metadata00.json" "${LOCAL_NODES_DIR}/metadata00.json"
cp "${SCRIPT_DIR}/config01.json"   "${LOCAL_NODES_DIR}/config01.json"
cp "${SCRIPT_DIR}/metadata01.json" "${LOCAL_NODES_DIR}/metadata01.json"

# 3. Generate a fresh ed25519 private key for each node.
echo "Generating private keys..."
for nn in 00 01; do
    if [[ "$(uname)" == "Linux" ]]; then
        openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    else
        /opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    fi
done

# 4. Prefetch expert models, Mistral IR3DE stats, and the Mistral tokenizer/embedder.
echo "Prefetching Hugging Face models, IR3DE stats, and Mistral embedder..."
python "${REPO_ROOT}/src/prefetch_example_models.py" \
    --repo-root "${REPO_ROOT}" \
    --default-stats "${REPO_ROOT}/ir3de_stats/default_stats.json" \
    "${SCRIPT_DIR}"/metadata*.json

echo
echo "Done. Run the two nodes from ${REPO_ROOT}, each in its own terminal:"
echo "  python run.py --peer-id 00"
echo "  python run.py --peer-id 01"
