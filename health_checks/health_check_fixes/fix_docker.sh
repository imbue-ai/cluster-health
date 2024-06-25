#!/bin/bash

set -ux

docker info && { docker info | grep Storage | grep -q overlay || exit 0 ; } 
docker ps | grep -v STATUS | awk '{ print $1 }' | xargs sudo docker kill
systemctl stop docker.socket
cp /etc/docker/daemon.json /tmp/
cat /tmp/daemon.json | jq ' ."storage-driver"="zfs"' > /etc/docker/daemon.json
if mount | grep -q '^overlay'; then
    mount | grep '^overlay' | awk '{ print $3 }' | xargs umount
else
    echo "No overlay filesystems are currently mounted."
fi
timeout 200 rm -r /var/lib/docker/overlay2/
systemctl restart docker.socket
docker info | grep Storage | grep -q zfs || exit 1