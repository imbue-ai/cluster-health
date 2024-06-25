#!/bin/bash

REMOTE_IP=$1
REMOTE_USER=$2
REMOTE_PORT=$3

execute_remote() {
    ssh {REMOTE_USER}@${REMOTE_IP} -p ${REMOTE_PORT} "$@"
}

execute_remote << 'EOF'
  if [ ! -d /tmp ]; then
   exit 1
  fi
EOF

if [ $? -ne 0 ]; then
    echo "Directory /tmp does not exist on ${REMOTE_IP}"
    exit 1
fi

script_dir=$(dirname "$(realpath "$0")")

scp -P ${REMOTE_PORT} ${script_dir}/fix_docker.sh ${REMOTE_USER}@${REMOTE_IP}:/tmp

echo "FINISHED COPYING FILES from ${script_dir} to ${REMOTE_IP}:/tmp"

# Pre-reboot commands
execute_remote << 'EOF'
    sudo bash tmp/fix_docker.sh
EOF
