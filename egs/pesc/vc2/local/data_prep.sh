#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=100
num_eval=100
num_train=932
train_set="train"
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
    echo "    --num_dev: number of development uttreances (default=100)."
    echo "    --num_eval: number of evaluation uttreances (default=100)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"

# make scp
find "$(realpath ${db_root})" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> "${scp}"
done

# split
utils/split_data.sh \
    --num_first 50 \
    --num_second 150 \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${eval_set}" \
    "${data_dir}/devtrain"
utils/split_data.sh \
    --num_first 10 \
    --num_second 140 \
    --shuffle "${shuffle}" \
    "${data_dir}/devtrain" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${train_set}"

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/devtrain"

echo "Successfully prepared data."
