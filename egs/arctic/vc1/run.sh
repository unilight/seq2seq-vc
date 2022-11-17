#!/usr/bin/env bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

conf=conf/vtn.v1.yaml

# dataset configuration
db_root=downloads
dumpdir=dump                # directory to dump full features
srcspk=clb                  # available speakers: "slt" "clb" "bdl" "rms"
trgspk=slt
num_train_utts=932
stats_ext=h5
norm_name=                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

# pretrained model related
pretrained_model=           # available pretrained models: m_ailabs.judy.vtn_tts_pt

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint 
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # The number of iterations of Griffin-Lim
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

pair=${srcspk}_${trgspk}
src_train_set=${srcspk}_train
src_dev_set=${srcspk}_dev
src_eval_set=${srcspk}_eval
trg_train_set=${trgspk}_train
trg_dev_set=${trgspk}_dev
trg_eval_set=${trgspk}_eval
pair_train_set=${pair}_train
pair_dev_set=${pair}_dev
pair_eval_set=${pair}_eval

                                       
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root} ${srcspk}
    local/data_download.sh ${db_root} ${trgspk}

    # download pretrained model for training
    #if [ -n "${pretrained_model}" ]; then
    #    local/pretrained_model_download.sh ${db_root} ${pretrained_model}
    #fi

    # download pretrained PWG
    if [ ${voc} == "PWG" ]; then
       local/pretrained_model_download.sh ${db_root} pwg_${trgspk}
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for spk in ${srcspk} ${trgspk}; do
        local/data_prep.sh \
            --train_set "${spk}_train" \
            --dev_set "${spk}_dev" \
            --eval_set "${spk}_eval" \
            "${db_root}/cmu_us_${spk}_arctic" "${spk}" data
    done
fi

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    pids=()
    for name in "${srcspk}_train" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train" "${trgspk}_dev" "${trgspk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            preprocess.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --segments "${dumpdir}/${name}/raw/segments.JOB" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # calculate statistics for normalization
    for name in "${srcspk}_train" "${trgspk}_train"; do
    (
        echo "Statistics computation start. See the progress via ${dumpdir}/${name}/compute_statistics.log."
        ${train_cmd} "${dumpdir}/${name}/compute_statistics.log" \
            compute_statistics.py \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}" \
                --verbose "${verbose}"
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished calculation of statistics."

    # normalize and dump them
    for spk in ${srcspk} ${trgspk}; do
        pids=()
        for name in "${spk}_train" "${spk}_dev" "${spk}_eval"; do
        (
            [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
            echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize.*.log."
            ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize.JOB.log" \
                normalize.py \
                    --config "${conf}" \
                    --stats "${dumpdir}/${spk}_train/stats.${stats_ext}" \
                    --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                    --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                    --verbose "${verbose}" \
                    --skip-wav-copy
            echo "Successfully finished normalization of ${name} set."
        ) &
        pids+=($!)
        done
        i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
        [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
        echo "Successfully finished ${spk} side normalization."
    done
fi


if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${srcspk}_train/stats.${stats_ext}" "${expdir}/${srcspk}_stats.${stats_ext}"
    cp "${dumpdir}/${trgspk}_train/stats.${stats_ext}" "${expdir}/${trgspk}_stats.${stats_ext}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet."
        # train="python -m seq2seq_vc.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="vc_train.py"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --src-train-dumpdir "${dumpdir}/${srcspk}_train/norm_${norm_name}" \
            --src-dev-dumpdir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
            --trg-train-dumpdir "${dumpdir}/${trgspk}_train/norm_${norm_name}" \
            --trg-dev-dumpdir "${dumpdir}/${trgspk}_dev/norm_${norm_name}" \
            --trg-stats "${expdir}/${trgspk}_stats.${stats_ext}" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            vc_decode.py \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}" \
                --checkpoint "${checkpoint}" \
                --trg-stats "${expdir}/${trgspk}_stats.${stats_ext}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    done
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        wavdir="${outdir}/${name}/wav"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir ${wavdir} \
                --data_root "${db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml"
    done
fi