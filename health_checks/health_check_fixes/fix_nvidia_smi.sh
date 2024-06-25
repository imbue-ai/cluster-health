#!/bin/bash

REMOTE_IP=$1
REMOTE_USER=$2
REMOTE_PORT=$3
execute_remote() {
    ssh ${REMOTE_USER}@${REMOTE_IP} -p ${REMOTE_PORT} "$@"
}

# Pre-reboot commands
execute_remote << 'EOF'
     nvidia-smi -e 1
EOF

# Reboot the machine
echo "Rebooting ${REMOTE_IP}..."
execute_remote "sudo reboot -h now"

# Wait for machine to reboot
while true; do
    if execute_remote "exit"; then
        break
    fi
    sleep 10  # Wait for 10 seconds before trying again
done
