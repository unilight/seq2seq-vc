#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=48000
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/cms_us_slt_arctic slt data"
    echo ""
    echo "Options:"
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    exit 1
fi

set -euo pipefail

for set_name in ${train_set} ${dev_set} ${eval_set}; do

    [ ! -e "${data_dir}/${spk}_${set_name}" ] && mkdir -p "${data_dir}/${spk}_${set_name}"

    # set filenames
    scp="${data_dir}/${spk}_${set_name}/wav.scp"

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"

    # make scp
    find "$(realpath ${db_root})/wav/${set_name}" -name "*.wav" -follow | sort | while read -r filename; do
        id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
    done

    echo "Successfully prepared ${spk} ${set_name} data."

done
