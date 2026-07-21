#!/usr/bin/env bash
#
# Installs example 3 (see README.md in this directory): backs up whatever config*.json/
# metadata*.json currently sit in local_nodes/ root, copies this example's
# own config00..07.json/metadata00..07.json there, and generates a fresh
# ed25519 private key for each of the eight nodes.
#
# This example also needs ir3de_stats/default_stats.json to serve NO stats
# by default, so that each node's own metadata.json "stats" list is the
# only thing it serves (see README.md's IR3DE stats section for why). This
# script backs that file up alongside everything else and replaces it with
# an empty one; RESTORE_DEFAULT_STATS.md (written into the backup directory)
# has the one-line command to put the original back once you're done.
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

# 2. Back up the real ir3de_stats/default_stats.json into the same backup
#    directory, then replace it with one that serves no default stats.
mkdir -p "${BACKUP_DIR}"
echo "Backing up ir3de_stats/default_stats.json to ${BACKUP_DIR}..."
cp "${REPO_ROOT}/ir3de_stats/default_stats.json" "${BACKUP_DIR}/default_stats.json"
cat > "${REPO_ROOT}/ir3de_stats/default_stats.json" << 'EOF'
{
  "stats": []
}
EOF
cat > "${BACKUP_DIR}/RESTORE_DEFAULT_STATS.md" << EOF
Restore the original default_stats.json (that example_3's setup.sh replaced) with:

  cp "${BACKUP_DIR}/default_stats.json" "${REPO_ROOT}/ir3de_stats/default_stats.json"
EOF

# 3. Install this example's config/metadata files.
echo "Installing example config/metadata into ${LOCAL_NODES_DIR}..."
for nn in "${NODE_IDS[@]}"; do
    cp "${SCRIPT_DIR}/config${nn}.json"   "${LOCAL_NODES_DIR}/config${nn}.json"
    cp "${SCRIPT_DIR}/metadata${nn}.json" "${LOCAL_NODES_DIR}/metadata${nn}.json"
done

# 4. Generate a fresh ed25519 private key for each node.
echo "Generating private keys..."
for nn in "${NODE_IDS[@]}"; do
    if [[ "$(uname)" == "Linux" ]]; then
        openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    else
        /opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk${nn}.pem"
    fi
done

echo
echo "Done. Run the eight nodes from ${REPO_ROOT}, each in its own terminal:"
for nn in "${NODE_IDS[@]}"; do
    echo "  python run.py --peer-id ${nn}"
done
echo
echo "IMPORTANT: ir3de_stats/default_stats.json was replaced with an empty one for this"
echo "example. When you're done, restore the original with:"
echo "  cp \"${BACKUP_DIR}/default_stats.json\" \"${REPO_ROOT}/ir3de_stats/default_stats.json\""
