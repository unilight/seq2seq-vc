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

conf=conf/vtn.tts_pt.v1.yaml

# dataset configuration
arctic_db_root=../../arctic/vc1/downloads # default saved here
db_root=/data/group1/z44476r/Corpora/l2-arctic # PLEASE CHANGE THIS
dumpdir=dump                # directory to dump full features
srcspk=TXHC                 # available speakers: "slt" "clb" "bdl" "rms"
trgspk=bdl
num_train=1032
stats_ext=h5
norm_name=ljspeech                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization
feat_type=mel

# pretrained model related
pretrained_model_checkpoint=downloads/ljspeech/transformer_tts_aept/checkpoint-50000steps.pkl
npvc_checkpoint=downloads/s3prl-vc-ppg_sxliu/checkpoint-50000steps.pkl
npvc_name=ppg_sxliu

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

# sanity check for norm_name and pretrained_model_checkpoint
if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
elif [ ${norm_name} == "self" ]; then
    echo "Please use pre-training."
else
    if [ -z ${pretrained_model_checkpoint} ]; then
        echo "Please specify the pretrained model checkpoint."
        exit 1
    fi
    pretrained_model_dir="$(dirname ${pretrained_model_checkpoint})"
    stats="${pretrained_model_dir}/stats.${stats_ext}"
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    # download ARCTIC
    ../../arctic/vc1/local/data_download.sh ${arctic_db_root} ${trgspk}

    # download pretrained vocoder
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_bdl/checkpoint-400000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_bdl/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_bdl/stats.h5"

    # download npvc model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/stats.h5"

    # download pretrained seq2seq model
    utils/hf_download.py --repo_id "unilight/seq2seq-vc" --outdir "downloads" --filename "ljspeech/transformer_tts_aept/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/seq2seq-vc" --outdir "downloads" --filename "ljspeech/transformer_tts_aept/config.yml"
    utils/hf_download.py --repo_id "unilight/seq2seq-vc" --outdir "downloads" --filename "ljspeech/transformer_tts_aept/stats.h5"
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "Preparing target speaker ${trgspk}"
    ../../arctic/vc1/local/data_prep.sh \
        --train_set "${trgspk}_train_${num_train}" \
        --dev_set "${trgspk}_dev" \
        --eval_set "${trgspk}_eval" \
        --num_train ${num_train} \
        --num_dev 50 --num_eval 50 \
        "${arctic_db_root}/cmu_us_${trgspk}_arctic" "${trgspk}" data

    echo "Preparing source speaker ${srcspk}"
    local/data_prep.sh \
        --train_set "${srcspk}_train_${num_train}" \
        --dev_set "${srcspk}_dev" \
        --eval_set "${srcspk}_eval" \
        --num_train ${num_train} \
        --num_dev 50 --num_eval 50 \
        "${db_root}/${srcspk}" "${srcspk}" data
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
    for name in "${srcspk}_train_${num_train}" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train_${num_train}" "${trgspk}_dev" "${trgspk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            preprocess.py \
                --config "${config_for_feature_extraction}" \
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
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Statistics computation (optional) and normalization"

    if [ ${norm_name} == "self" ]; then
        # calculate statistics for normalization
        for name in "${srcspk}_train_${num_train}" "${trgspk}_train_${num_train}"; do
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
    fi

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # normalize and dump them
    for spk in ${srcspk} ${trgspk}; do
        pids=()
        for name in "${spk}_train_${num_train}" "${spk}_dev" "${spk}_eval"; do
        (
            [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
            echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize.*.log."
            ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize.JOB.log" \
                normalize.py \
                    --config "${config_for_feature_extraction}" \
                    --stats "${stats}" \
                    --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                    --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                    --verbose "${verbose}" \
                    --feat_type "${feat_type}" \
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
    expname=${srcspk}_${trgspk}_${num_train}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}_${num_train}_${tag}
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
        pretrained_model_checkpoint_name=$(basename ${pretrained_model_checkpoint%.*})
        cp "${pretrained_model_dir}/stats.${stats_ext}" "${expdir}/"
        cp "${pretrained_model_dir}/config.yml" "${expdir}/original_config.yml"
        cp "${pretrained_model_checkpoint}" "${expdir}/original_${pretrained_model_checkpoint_name}.pkl"
        echo "Training start. See the progress via ${expdir}/train.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
            vc_train.py \
                --config "${expdir}/original_config.yml" \
                --additional-config "${conf}" \
                --src-train-dumpdir "${dumpdir}/${srcspk}_train_${num_train}/norm_${norm_name}" \
                --src-dev-dumpdir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
                --src-feat-type "${feat_type}" \
                --trg-train-dumpdir "${dumpdir}/${trgspk}_train_${num_train}/norm_${norm_name}" \
                --trg-dev-dumpdir "${dumpdir}/${trgspk}_dev/norm_${norm_name}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --trg-feat-type "${feat_type}" \
                --init-checkpoint "${expdir}/original_${pretrained_model_checkpoint_name}.pkl" \
                --outdir "${expdir}" \
                --resume "${resume}" \
                --verbose "${verbose}"
    else
        echo "Please use a pre-trained seq2seq model."
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
                --checkpoint "${checkpoint}" \
                --src-feat-type "${feat_type}" \
                --trg-feat-type "${feat_type}" \
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
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir "${outdir}/${name}" \
                --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --asr --mcd
        grep "Mean MCD" "${outdir}/${name}/evaluation.log"
        grep "Mean CER" "${outdir}/${name}/evaluation.log"
    done
fi



if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "Stage 6: Stage 2 decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"

    npvc_pretrained_model_dir=$(dirname ${npvc_checkpoint})
    npvc_stats="${npvc_pretrained_model_dir}/stats.h5"
    echo "NPVC pretrained model checkpoint: ${npvc_checkpoint}"

    pids=()
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
    (
        new_name="stage2_${npvc_name}_$(basename "${npvc_checkpoint}" .pkl)_${name}"
        [ ! -e "${outdir}/${new_name}" ] && mkdir -p "${outdir}/${new_name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${new_name}/decode.*.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${new_name}/decode.log" \
            s3prl-vc-decode \
                --wavdir "${outdir}/${name}" \
                --checkpoint "${npvc_checkpoint}" \
                --trg-stats "${npvc_stats}" \
                --outdir "${outdir}/${new_name}/" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Stage 2 Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        new_name="stage2_${npvc_name}_$(basename "${npvc_checkpoint}" .pkl)_${name}"
        echo "Evaluation start. See the progress via ${outdir}/${new_name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${new_name}/evaluation.log" \
            local/evaluate.py \
                --wavdir "${outdir}/${new_name}" \
                --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --asr
        grep "Mean CER" "${outdir}/${new_name}/evaluation.log"
    done
fi

################################################################################
###################################  LEGACY  ###################################
################################################################################

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Ground Truth Objective Evaluation"

    echo "Evaluation start. See the progress via ${srcspk}_gt_evaluation.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${srcspk}_gt_evaluation.log" \
        local/gt_evaluate.py \
            --wavdir "${db_root}/${srcspk}/wav" \
            --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
            --trgspk ${trgspk} \
            --f0_path "conf/f0.yaml"
    grep "CER:" "${srcspk}_gt_evaluation.log"

    echo "Evaluation start. See the progress via ${trgspk}_gt_evaluation.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${trgspk}_gt_evaluation.log" \
        local/gt_evaluate.py \
            --wavdir "${arctic_db_root}/cmu_us_${trgspk}_arctic/wav" \
            --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
            --trgspk ${trgspk} \
            --f0_path "conf/f0.yaml"
    grep "CER:" "${trgspk}_gt_evaluation.log"
fi