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
 ./axl/node -config "local_nodes/c${node_id}.json"
