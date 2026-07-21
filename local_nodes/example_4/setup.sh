#!/usr/bin/env bash
#
# Installs example 4 (see README.md in this directory) on THIS machine, playing whichever role (A/B/C) is
# passed as an argument. Run once per physical machine, with a different
# role on each of the three:
#
#   ./local_nodes/example_4/setup.sh A     # on the Quetzalcoatl machine
#   ./local_nodes/example_4/setup.sh B     # on the Tezcatlipoca machine
#   ./local_nodes/example_4/setup.sh C     # on the Huitzilopochtli machine
#
# Backs up whatever local_nodes/config.json, local_nodes/metadata.json, and
# local_nodes/pk.pem currently exist, installs this example's config.json
# (same file for every role — it's a placeholder-based template, see the
# README) and the role-specific metadata<ROLE>.json as local_nodes/metadata.json,
# and generates a fresh ed25519 private key. You still need to edit
# local_nodes/config.json by hand afterward to fill in the real IP address
# of the next node in the cycle (see the README).

set -euo pipefail

ROLE="${1:-}"
if [[ ! "${ROLE}" =~ ^[ABCabc]$ ]]; then
    echo "Usage: $0 <A|B|C>" >&2
    exit 1
fi
ROLE="$(echo "${ROLE}" | tr '[:lower:]' '[:upper:]')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_NODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_NODES_DIR}/.." && pwd)"

# 1. Back up whatever's currently in local_nodes/ root for the default
#    (no-suffix) local peer — reversible, matches the existing
#    YYYYMMDD-bkp convention already used in this directory.
BACKUP_DIR="${LOCAL_NODES_DIR}/$(date +%Y%m%d-%H%M%S)-bkp"
existing=()
for f in config.json metadata.json pk.pem; do
    [[ -e "${LOCAL_NODES_DIR}/${f}" ]] && existing+=("${LOCAL_NODES_DIR}/${f}")
done
if [[ ${#existing[@]} -gt 0 ]]; then
    echo "Backing up ${#existing[@]} existing file(s) to ${BACKUP_DIR}..."
    mkdir -p "${BACKUP_DIR}"
    mv "${existing[@]}" "${BACKUP_DIR}/"
fi

# 2. Install this example's config.json (identical for every role) and the
#    role-specific metadata file.
echo "Installing config.json and metadata${ROLE}.json (role ${ROLE}) into ${LOCAL_NODES_DIR}..."
cp "${SCRIPT_DIR}/config.json"        "${LOCAL_NODES_DIR}/config.json"
cp "${SCRIPT_DIR}/metadata${ROLE}.json" "${LOCAL_NODES_DIR}/metadata.json"

# 3. Generate a fresh ed25519 private key for this node.
echo "Generating private key..."
if [[ "$(uname)" == "Linux" ]]; then
    openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk.pem"
else
    /opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${LOCAL_NODES_DIR}/pk.pem"
fi

echo
echo "Done. Before running this node:"
echo "  1. Edit ${LOCAL_NODES_DIR}/config.json and replace <NEXT_NODE_IP> with"
echo "     the real, reachable IP address of the next node in the cycle"
echo "     (see the README's Topology section for which role that is)."
echo "  2. Make sure this machine's port 9000 is reachable from that setup"
echo "     (firewall/NAT permitting)."
echo
echo "Then, from ${REPO_ROOT}:"
echo "  python run.py"
