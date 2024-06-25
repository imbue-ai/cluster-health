#!/bin/bash

REMOTE_IP=$1
REMOTE_USER=$2
REMOTE_PORT=$3
execute_remote() {
    ssh ${REMOTE_USER}@${REMOTE_IP} -p ${REMOTE_PORT} "$@"
}

execute_remote << 'EOF'
     echo "Pre-reboot code"
EOF

script_dir=$(dirname "$(realpath "$0")")

scp -P ${REMOTE_PORT} ${script_dir}/uninstall_nvidia.sh ${REMOTE_USER}@${REMOTE_IP}:/tmp
scp -P ${REMOTE_PORT} ${script_dir}/reinstall_nvidia.sh ${REMOTE_USER}@${REMOTE_IP}:/tmp

echo "FINISHED COPYING FILES from ${script_dir} to ${REMOTE_IP}:/tmp"

# Pre-reboot commands
execute_remote << 'EOF'
     bash /tmp/uninstall_nvidia.sh
     bash /tmp/reinstall_nvidia.sh
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

# Post-reboot commands
echo "${REMOTE_IP} reboot completed"
execute_remote << 'EOF'
     echo "Post-reboot code"
EOF
