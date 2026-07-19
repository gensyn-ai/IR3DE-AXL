#!/usr/bin/env bash

if [[ "$1" =~ ^[0-9]+$ ]]; then
    node_id="$(printf '%02d' "$1")"
    exec ./axl/node -config "local_nodes/config${node_id}.json"
else
    exec ./axl/node -config "local_nodes/config.json"
fi
