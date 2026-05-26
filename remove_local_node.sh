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
nodes_dir="local_nodes"
json_file="${nodes_dir}/config${node_id}.json"
pem_file="${nodes_dir}/pk${node_id}.pem"

if [[ ! -f "${json_file}" ]]; then
    echo "Error: node file ${json_file} does not exist." >&2
    exit 1
fi

echo "Removing node file ${json_file}..."
rm "${json_file}"

if [[ -f "${pem_file}" ]]; then
    echo "Removing private key file ${pem_file}..."
    rm "${pem_file}"
else
    echo "Warning: private key file ${pem_file} does not exist." >&2
fi
echo "Node config${node_id} removed successfully."
