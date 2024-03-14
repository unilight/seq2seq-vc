#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# This recipe is almost completely based on the original Urhythimc codebase.
# https://github.com/bshall/urhythmic

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

conf=conf/vtn.v1.yaml       # this is only used for data io

# dataset configuration
db_root=../vc1/downloads    # change this to `downloads` if you did not run vc1
dumpdir=dump                # directory to dump full features
srcspk=clb                  # available speakers: "clb" "bdl"
trgspk=slt                  # available speakers: "slt" "rms"
num_train=932

# evaluation related setting
gv=False
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root} ${srcspk}
    local/data_download.sh ${db_root} ${trgspk}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for spk in ${srcspk} ${trgspk}; do
        echo "Data prep for ${spk}"
        local/data_prep.sh \
            --train_set "${spk}_train_${num_train}" \
            --dev_set "${spk}_dev" \
            --eval_set "${spk}_eval" \
            --num_train ${num_train} \
            "${db_root}/cmu_us_${spk}_arctic" "${spk}" data
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Extract Soft Speech Units and Log Probabilities"

    for name in "${srcspk}_train_${num_train}" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train_${num_train}" "${trgspk}_dev" "${trgspk}_eval"; do
        [ ! -e "${dumpdir}/${name}" ] && mkdir -p "${dumpdir}/${name}"
        echo "Extraction start. See the progress via ${dumpdir}/${name}/soft_extraction.log."
        ${train_cmd} "${dumpdir}/${name}/soft_extraction.log" \
            urhythmic_encode.py \
                --config "${conf}" \
                --scp "data/${name}/wav.scp" \
                --segments "data/${name}/segments" \
                --dumpdir "${dumpdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    done
    echo "Successfully finished feature extraction."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Segmenter Training"

    echo "Not Implemented. Use pre-trained segmenter for now."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Segmentation"

    for name in "${srcspk}_train_${num_train}" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train_${num_train}" "${trgspk}_dev" "${trgspk}_eval"; do
        [ ! -e "${dumpdir}/${name}/segments" ] && mkdir -p "${dumpdir}/${name}/segments"
        echo "Segmentation start. See the progress via ${dumpdir}/${name}/segmentation.log."
        ${train_cmd} "${dumpdir}/${name}/segmentation.log" \
            urhythmic_segment.py \
                --data_dir "${dumpdir}/${name}/logprobs" \
                --dumpdir "${dumpdir}/${name}/segments" \
                --verbose "${verbose}"
        echo "Successfully finished segmentation of ${name} set."
    done
    echo "Successfully finished segmentation."
fi

expdir="exp/${srcspk}_${trgspk}_${num_train}"
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Rhythm model training"

    for spk in "${srcspk}" "${trgspk}"; do
        [ ! -e "${expdir}/rhythm_model" ] && mkdir -p "${expdir}/rhythm_model"
        echo "Rhythm model training for ${spk} start. See the progress via ${expdir}/rhythm_model/rhythm_model_train_${spk}.log."
        ${train_cmd} "${expdir}/rhythm_model/rhythm_model_train_${spk}.log" \
            urhythmic_train_rhythm_model.py \
                --data_dir "${dumpdir}/${spk}_train_${num_train}/segments" \
                --checkpoint_path "${expdir}/rhythm_model/rhythm_fine_${spk}.pt" \
                --verbose "${verbose}"
        echo "Successfully finished rhythm model training for ${spk}."
    done
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Vocoder fine-tuning"

        # download pretrained vocoder checkpoint if not exist
    [ ! -e "${pretrained_vocoder_checkpoint}" ] && wget https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-LJSpeech-ceb1368d.pt -O "${pretrained_vocoder_checkpoint}"

    [ ! -e "${expdir}/vocoder" ] && mkdir -p "${expdir}/vocoder"
    echo "Training start. See the progress via ${expdir}/vocoder/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/vocoder/train.log" \
        urhythmic_fine_tune_vocoder.py \
            --train_unit_dir "${dumpdir}/${trgspk}_train_${num_train}/soft" \
            --dev_unit_dir "${dumpdir}/${trgspk}_dev/soft" \
            --train_wav_scp "data/${trgspk}_train_${num_train}/wav.scp" \
            --dev_wav_scp "data/${trgspk}_dev/wav.scp" \
            --checkpoint_dir "${expdir}/vocoder/" \
            --resume "${pretrained_vocoder_checkpoint}"

fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "Stage 6: Conversion"

    vocoder_path="$(ls -dt "${expdir}"/vocoder/*.pt | head -1 || true)"

    outdir="${expdir}/results"
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        [ ! -e "${outdir}/${name}/wav" ] && mkdir -p "${outdir}/${name}/wav"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Conversion start. See the progress via ${outdir}/${name}/decode.log."
        CUDA_VISIBLE_DEVICES="" ${cuda_cmd} --gpu 0 "${outdir}/${name}/decode.log" \
            urhythmic_convert.py \
                --config "${conf}" \
                --scp "data/${name}/wav.scp" \
                --segments "data/${name}/segments" \
                --src_rhythm_model_path "${expdir}/rhythm_model/rhythm_fine_${srcspk}.pt" \
                --trg_rhythm_model_path "${expdir}/rhythm_model/rhythm_fine_${trgspk}.pt" \
                --vocoder_path "${vocoder_path}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    done
    echo "Successfully finished conversion."
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation"

    outdir="${expdir}/results"
    for _set in "dev" "eval"; do
        name="${srcspk}_${_set}"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir "${outdir}/${name}" \
                --data_root "${db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --segments "data/${trgspk}_${_set}/segments" \
                --gv ${gv}
        grep "Mean MCD" "${outdir}/${name}/evaluation.log"
    done
fi