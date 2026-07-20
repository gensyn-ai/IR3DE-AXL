#!/usr/bin/env bash
#
# Utility for simulating custom node topologies. Creates a new simulated
# local node: picks the first free node id (00-99), generates its ed25519
# private key, randomly wires it up to some existing local nodes as peers,
# and writes its local_nodes/configNN.json and metadataNN.json. There's no
# need to run this to try the app (pre-built example node setups are
# provided separately) — use it if you want to simulate a different graph
# of nodes than what's provided.
#
# The generated metadataNN.json is almost empty (just node_id/node_name) —
# edit it by hand afterward to add "models" and/or "stats" entries (see
# Peer.get_models_info/get_stats_info for the expected shape) if you want
# this node to actually serve experts or IR3DE stats.
#
# Usage:
#   ./scripts/add_local_node.sh [NUM_PEERS] [NODE_NAME]
#
# NUM_PEERS (default 1) is how many existing local nodes to connect the new
# one to (capped at however many already exist). NODE_NAME (default
# "Node NN") is a display-only name stored in its metadata.

num_peers="${1:-1}"

nodes_dir="local_nodes"
if [[ ! -d "${nodes_dir}" ]]; then
	echo "Creating directory ${nodes_dir} for local nodes..."
	mkdir -p "${nodes_dir}"
fi

# FINDING FIRST MISSING NODE ID
echo "Looking for the first missing node file in ${nodes_dir}..."
missing_nn=""

for n in $(seq -w 0 99); do
	json_file="${nodes_dir}/config${n}.json"
	if [[ ! -f "${json_file}" ]]; then
		missing_nn="${n}"
		break
	fi
done

if [[ -z "${missing_nn}" ]]; then
	echo "Error: all nodes from 0 to 99 already exist." >&2
	exit 1
fi
echo "First missing node: ${missing_nn}."

node_name="${2:-"Node ${missing_nn}"}"

# GENERATE PRIVATE KEY
echo "Generating private key for node ${missing_nn}..."
if [[ -f "${nodes_dir}/pk${missing_nn}.pem" ]]; then
    echo "private key file ${nodes_dir}/pk${missing_nn}.pem already exists, deleting and generating a new one."
    rm "${nodes_dir}/pk${missing_nn}.pem"
fi

if [[ "$(uname)" == "Linux" ]]; then
    openssl genpkey -algorithm ed25519 -out "${nodes_dir}/pk${missing_nn}.pem"
else
    /opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${nodes_dir}/pk${missing_nn}.pem"
fi

# BUILDING PEER LIST FROM EXISTING NODES
candidate_ids=()
for cfg in "${nodes_dir}"/config[0-9][0-9].json; do
	[[ -e "${cfg}" ]] || continue
	id="${cfg##*config}"
	id="${id%.json}"
	if [[ "${id}" != "${missing_nn}" ]]; then
		candidate_ids+=("${id}")
	fi
done

available_peers=${#candidate_ids[@]}
echo "Found ${available_peers} existing peer(s) to connect to: ${candidate_ids[*]}"

# VALIDATE num_peers
if (( num_peers < 1 && available_peers > 0 )); then
	num_peers=1
	echo "num_peers must be at least 1 if available_peers > 0. Defaulting to 1 peer." >&2
fi
if (( num_peers > available_peers )); then
	num_peers=${available_peers}
	echo "num_peers cannot exceed the number of available peers (${available_peers}). Defaulting to ${available_peers} peer(s)." >&2
fi
if [[ ! "${num_peers}" =~ ^[0-9]+$ ]]; then
	echo "Error: num_peers must be a non-negative integer." >&2
	exit 1
fi


# SELECTING PEERS RANDOMLY
echo "Selecting ${num_peers} peer(s) to connect to..."

for ((i=${#candidate_ids[@]}-1; i>0; i--)); do
	j=$((RANDOM % (i + 1)))
	tmp="${candidate_ids[i]}"
	candidate_ids[i]="${candidate_ids[j]}"
	candidate_ids[j]="${tmp}"
done

echo "Randomly selected peers: ${candidate_ids[*]}"

# BUILDING PEERS JSON
echo "Building peers JSON array..."
peers_json="[]"
if (( num_peers > 0 )); then
	peers_json="["
	for ((i=0; i<num_peers; i++)); do
		peers_json+="\"tls://127.0.0.1:90${candidate_ids[i]}\""
		if (( i < num_peers - 1 )); then
			peers_json+=", "
		fi
	done
	peers_json+="]"
fi

echo "Selected ${num_peers} peer(s): ${peers_json}"

# CREATING NEW CONFIG FILE
echo "Creating new config file for node ${missing_nn}..."
new_file="${nodes_dir}/config${missing_nn}.json"
listen_port="90${missing_nn}"
api_port="91${missing_nn}"

cat > "${new_file}" << EOF
{
  "PrivateKeyPath": "${nodes_dir}/pk${missing_nn}.pem",
  "Peers": ${peers_json},
  "Listen": ["tls://127.0.0.1:${listen_port}"],
  "api_port": ${api_port}
}
EOF

echo "Created ${new_file}."

# CREATING METADATA FILE
echo "Creating metadata file for node ${missing_nn}..."
metadata_file="${nodes_dir}/metadata${missing_nn}.json"

cat > "${metadata_file}" << EOF
{
  "node_id": "${missing_nn}",
  "node_name": "${node_name}"
}
EOF
echo "Created ${metadata_file}."
