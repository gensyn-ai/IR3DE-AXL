#!/usr/bin/env bash
#
# Starts the AXL P2P node binary (./axl/node) for one local peer, pointed at
# its config file under local_nodes/. Not normally run by hand — peer.py
# launches it as a subprocess (Peer.__init__) using the same node id it was
# given. Config files must already exist (see scripts/add_local_node.sh).
#
# Usage:
#   ./scripts/start_node.sh [PEER_ID]
#
# PEER_ID is optional: with it, uses local_nodes/configNN.json (zero-padded
# to 2 digits); without it, uses local_nodes/config.json (the default local
# peer).

if [[ "$1" =~ ^[0-9]+$ ]]; then
    node_id="$(printf '%02d' "$1")"
    exec ./axl/node -config "local_nodes/config${node_id}.json"
else
    exec ./axl/node -config "local_nodes/config.json"
fi
