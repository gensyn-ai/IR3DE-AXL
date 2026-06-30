#!/usr/bin/env bash

node_id="${1:-}"

if [[ -z "${node_id}" ]]; then
    echo "Error: no node ID provided." >&2
    exit 1
fi

if [[ ! "${node_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: node ID must be numeric." >&2
    exit 1
fi

node_id="$(printf '%02d' "${node_id}")"
# Use exec so this script's PID becomes the node process itself. Without it, the
# Popen handle in Peer would point at this bash wrapper and SIGTERM/SIGKILL would
# orphan the node instead of stopping it.
exec ./axl/node -config "local_nodes/config${node_id}.json"
