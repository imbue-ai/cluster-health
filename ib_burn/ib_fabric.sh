#!/bin/bash
set -e
set -u

SOURCE="${BASH_SOURCE[0]}"
CONFIG="./config.json"

IB_SERVER_LIST="./ib_server.list"
IB_SWITCH_LIST="./ib_switch.list"
IB_SWITCH_LINK="./ib_switch.link"

cd $(dirname "$SOURCE")

echo > "$IB_SERVER_LIST"

function read_config {
    jq --raw-output --arg ARG "${2:-}" "$1" "$CONFIG"
}

SSH_USER=$(read_config '.node_info.user')
SSH_PORT=$(read_config '.node_indo.port')

for SSH_HOST in $(read_config '.node_info.nodes | keys[]')
do
    SSH_ADDR=$(read_config '.node_info.nodes[$ARG]' "$SSH_HOST")
    for IB_HCA in $(read_config '.ib_hcas | keys[]')
    do
        IB_GUID=$(ssh "$SSH_USER@$SSH_ADDR" "ibstat --short $IB_HCA | grep Node | tr ' ' '\t' | cut -f 4")
        echo "$IB_GUID $SSH_HOST $IB_HCA" >> "$IB_SERVER_LIST"
    done
done

ssh "$SSH_USER@$SSH_ADDR" 'ibswitches' | tr -d '"' | awk '{ print($3, $6) }' > "$IB_SWITCH_LIST"
ssh "$SSH_USER@$SSH_ADDR" 'iblinkinfo --line --switches-only' | grep 'Active/' \
    | sed -E 's/"[^"]*"//g;s/\([^)]+\)//g;s/\[[^]]*\]//g' | awk '{ print($1, $5, $3, $7) }' > "$IB_SWITCH_LINK"
