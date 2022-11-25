#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

num_dev=250
num_eval=250
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
cleaner=tacotron
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db> <data_dir>"
    exit 1
fi

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"
text="${data_dir}/all/text"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp
find "$(realpath ${db})" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> "${scp}"
done

# make text
# make text using the original text
paste -d " " \
    <(cut -d "|" -f 1 < ${db}/metadata.csv) \
    <(cut -d "|" -f 3 < ${db}/metadata.csv) \
    > ${text}
echo "finished making text."

# check
diff -q <(awk '{print $1}' "${scp}") <(awk '{print $1}' "${text}") > /dev/null

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
utils/split_data.sh \
    --num_first "${num_train}" \
    --num_second "${num_deveval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}" \
    "${data_dir}/deveval"
utils/split_data.sh \
    --num_first "${num_dev}" \
    --num_second "${num_eval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/deveval" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${eval_set}"

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/deveval"

echo "Successfully prepared data."
