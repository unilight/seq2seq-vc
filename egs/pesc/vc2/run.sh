#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction

conf=conf/aas_vc.melmelmel.v1.yaml

# dataset configuration
db_root=downloads
dumpdir=dump                # directory to dump full features
srcspk=EL_PS_FEMALE001
trgspk=SP_PS_FEMALE001
stats_ext=h5
norm_name=self                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

src_feat=mel
trg_feat=mel
dp_feat=mel

train_duration_dir=none     # need to be properly set if FS2-VC is used
dev_duration_dir=none       # need to be properly set if FS2-VC is used

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# evaluation related setting
gv=False                    # whether to calculate GV for evaluation
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

# sanity check for norm_name and pretrained_model_checkpoint
src_stats="${dumpdir}/${srcspk}_train/stats.${stats_ext}"
trg_stats="${dumpdir}/${trgspk}_train/stats.${stats_ext}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Dataset and Pretrained Model Download"

    # download dataset
    local/data_download.sh "${db_root}"

    # download ParallelWaveGAN model
    mkdir -p downloads/pwg
    utils/hf_download.py --repo_id "unilight/pesc-pwg" --outdir "downloads/pwg" --filename "checkpoint-400000steps.pkl"
    utils/hf_download.py --repo_id "unilight/pesc-pwg" --outdir "downloads/pwg" --filename "config.yml"
    utils/hf_download.py --repo_id "unilight/pesc-pwg" --outdir "downloads/pwg" --filename "stats.h5"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    # srcspk
    local/data_prep.sh \
        --train_set "${srcspk}_train" \
        --dev_set "${srcspk}_dev" \
        --eval_set "${srcspk}_eval" \
        "${db_root}/data/EL/${srcspk}" "${srcspk}" data
    ${train_cmd} "data/${srcspk}_train/create_histogram.log" \
        create_histogram.py \
            --scp "data/${srcspk}_train/wav.scp" \
            --figure_dir "data/${srcspk}_train"
    
    # trgspk
    local/data_prep.sh \
        --train_set "${trgspk}_train" \
        --dev_set "${trgspk}_dev" \
        --eval_set "${trgspk}_eval" \
        "${db_root}/data/SP/${trgspk}" "${trgspk}" data
    ${train_cmd} "data/${trgspk}_train/create_histogram.log" \
        create_histogram.py \
            --scp "data/${trgspk}_train/wav.scp" \
            --figure_dir "data/${trgspk}_train"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # extract raw features
    pids=()
    for name in "${srcspk}_train" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train" "${trgspk}_dev" "${trgspk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            preprocess.py \
                --config "${config_for_feature_extraction}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Statistics computation (optional) and normalization"

    if [ ${norm_name} == "self" ]; then
        # calculate statistics for normalization

        # src
        name="${srcspk}_train"
        echo "Statistics computation start. See the progress via ${dumpdir}/${name}/compute_statistics_${src_feat}.log."
        ${train_cmd} "${dumpdir}/${name}/compute_statistics_${src_feat}.log" \
            compute_statistics.py \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}" \
                --feat_type "${src_feat}" \
                --verbose "${verbose}"

        # trg
        name="${trgspk}_train"
        echo "Statistics computation start. See the progress via ${dumpdir}/${name}/compute_statistics_${trg_feat}.log."
        ${train_cmd} "${dumpdir}/${name}/compute_statistics_${trg_feat}.log" \
            compute_statistics.py \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}" \
                --feat_type "${trg_feat}" \
                --verbose "${verbose}"
    fi

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # normalize and dump them
    # src
    spk="${srcspk}"
    for name in "${spk}_train" "${spk}_dev" "${spk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize_${src_feat}.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize_${src_feat}.JOB.log" \
            normalize.py \
                --config "${config_for_feature_extraction}" \
                --stats "${src_stats}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --verbose "${verbose}" \
                --feat_type "${src_feat}" \
                --skip-wav-copy
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished ${spk} side normalization."

    # trg
    spk="${trgspk}"
    for name in "${spk}_train" "${spk}_dev" "${spk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize_${trg_feat}.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize_${trg_feat}.JOB.log" \
            normalize.py \
                --config "${config_for_feature_extraction}" \
                --stats "${trg_stats}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --verbose "${verbose}" \
                --feat_type "${trg_feat}" \
                --skip-wav-copy
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished ${spk} side normalization."
fi

if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet. Usually VC training using arctic can be done with 1 GPU."
        exit 1
    fi

    if [ ! -z ${pretrained_model_checkpoint} ]; then
        echo "Pretraining not Implemented yet."
        exit 1
    else
        cp "${dumpdir}/${trgspk}_train/stats.${stats_ext}" "${expdir}/"
        echo "Training start. See the progress via ${expdir}/train.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
            vc_train.py \
                --config "${conf}" \
                --src-train-dumpdir "${dumpdir}/${srcspk}_train/norm_${norm_name}" \
                --src-dev-dumpdir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
                --src-feat-type "${src_feat}" \
                --trg-train-dumpdir "${dumpdir}/${trgspk}_train/norm_${norm_name}" \
                --trg-dev-dumpdir "${dumpdir}/${trgspk}_dev/norm_${norm_name}" \
                --trg-feat-type "${trg_feat}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --train-dp-input-dir "${dumpdir}/${srcspk}_train/norm_${norm_name}" \
                --dev-dp-input-dir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
                --train-duration-dir "${train_duration_dir}" \
                --dev-duration-dir "${dev_duration_dir}" \
                --outdir "${expdir}" \
                --resume "${resume}" \
                --verbose "${verbose}"
    fi
    echo "Successfully finished training."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        CUDA_VISIBLE_DEVICES="" ${cuda_cmd} JOB=1:${n_jobs} --gpu 0 "${outdir}/${name}/decode.JOB.log" \
            vc_decode.py \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --dp_input_dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --src-feat-type "${src_feat}" \
                --trg-feat-type "${trg_feat}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for _set in "dev" "eval"; do
        name="${srcspk}_${_set}"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir "${outdir}/${name}" \
                --data_root "${db_root}/data/SP/${trgspk}" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --text_path "${db_root}/text.csv"
        grep "Mean MCD" "${outdir}/${name}/evaluation.log"
    done
fi