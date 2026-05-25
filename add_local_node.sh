#!/usr/bin/env bash

nodes_dir="local_nodes"
echo "Looking for the first missing node file in ${nodes_dir}..."
missing_nn=""

for n in $(seq -w 1 99); do
	json_file="${nodes_dir}/c${n}.json"
	if [[ ! -f "${json_file}" ]]; then
		missing_nn="${n}"
		break
	fi
done

if [[ -z "${missing_nn}" ]]; then
	echo "Error: all files from c01.json to c99.json already exist." >&2
	exit 1
fi
echo "First missing node file: c${missing_nn}.json"

echo "Generating private key for node ${missing_nn}..."
if [[ -f "${nodes_dir}/k${missing_nn}.pem" ]]; then
    echo "private key file ${nodes_dir}/k${missing_nn}.pem already exists., deleting and generating a new one."
    rm "${nodes_dir}/k${missing_nn}.pem"
fi

/opt/homebrew/opt/openssl/bin/openssl genpkey -algorithm ed25519 -out "${nodes_dir}/k${missing_nn}.pem"

new_file="${nodes_dir}/c${missing_nn}.json"
listen_port="90${missing_nn}"
api_port="91${missing_nn}"

cat > "${new_file}" << EOF
{
  "PrivateKeyPath": "${nodes_dir}/k${missing_nn}.pem",
  "Peers": ["tls://127.0.0.1:9001"],
  "Listen": ["tls://127.0.0.1:${listen_port}"],
  "api_port": ${api_port}
}
EOF

echo "Created ${new_file}"
