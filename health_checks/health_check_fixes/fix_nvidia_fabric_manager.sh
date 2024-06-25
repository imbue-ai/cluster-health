#!/bin/bash

REMOTE_IP=$1
REMOTE_USER=$2
REMOTE_PORT=$3

execute_remote() {
    ssh ${REMOTE_USER}@${REMOTE_IP} -p ${REMOTE_PORT} "$@"
}

# Remote commands
execute_remote << 'EOF'
     sudo service nvidia-fabricmanager restart
EOF
