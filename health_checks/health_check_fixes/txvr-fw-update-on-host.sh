#!/bin/bash


FW='fw_47_171_00001_dev_signed.bin'

if [ "$1" = '--dry-run' ] ; then
        DRY_RUN=echo
else
        DRY_RUN=
fi

cd /tmp

if [[ ! -f $FW ]]
then
    wget "https://networkingdownloads.nvidia.com/custhelp/Non_Monetized_Products/LinkX/MMA4Z00/$FW"
fi

script_dir=$(dirname "$(realpath "$0")")

for HCA in $(jq --raw-output '.ib_hcas | keys[]' "${script_dir}"/../config.json) ;
do
    flint -d $HCA --linkx --downstream_device_ids 1 query full
    FV=$(flint -d $HCA --linkx --downstream_device_ids 1 query | grep 47.171.0001)
    if [[ $FV == "" ]]
    then
        $DRY_RUN flint -d $HCA --linkx --linkx_auto_update --image $FW --download_transfer --activate burn || echo 'burn failed ' $HCA
    fi
done
