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

conf=conf/vtn.tts_pt.v1.ppg_sxliu.yaml

# dataset configuration
arctic_db_root=../../arctic/vc1/downloads       # default saved here
db_root=/data/group1/z44476r/Corpora/l2-arctic  # PLEASE CHANGE THIS
dumpdir=dump                                    # directory to dump full features
srcspk=TXHC
trgspk=bdl
num_train=1032
stats_ext=h5
norm_name=ljspeech                              # used to specify normalized data.

# pretrained model related
pretrained_model_checkpoint=downloads/ljspeech_text_to_ppg_sxliu_aept/checkpoint-50000steps.pkl
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

npvc_pretrained_model_dir=$(dirname ${npvc_checkpoint})
npvc_stats="${npvc_pretrained_model_dir}/stats.h5"
npvc_config="${npvc_pretrained_model_dir}/config.yml"
echo "NPVC pretrained model checkpoint: ${npvc_checkpoint}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    # download ARCTIC
    ../../arctic/vc1/local/data_download.sh ${arctic_db_root} ${trgspk}

    # download pretrained vocoder
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/checkpoint-400000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/stats.h5"

    # download npvc model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/stats.h5"

    # download decoder model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ppg_sxliu_decoder_THXC/checkpoint-10000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ppg_sxliu_decoder_THXC/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ppg_sxliu_decoder_THXC/stats.h5"

    # download pretrained seq2seq model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_text_to_ppg_sxliu_aept/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_text_to_ppg_sxliu_aept/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_text_to_ppg_sxliu_aept/stats.h5"

    # download pretrained (ready to use) model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "exp/TXHC_bdl_1032_pretrained" --filename "lsc_THXC_ppg_sxliu/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "exp/TXHC_bdl_1032_pretrained" --filename "lsc_THXC_ppg_sxliu/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "exp/TXHC_bdl_1032_pretrained" --filename "lsc_THXC_ppg_sxliu/stats.h5"
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
    ../cascade/local/data_prep.sh \
        --train_set "${srcspk}_train_${num_train}" \
        --dev_set "${srcspk}_dev" \
        --eval_set "${srcspk}_eval" \
        --num_train ${num_train} \
        --num_dev 50 --num_eval 50 \
        "${db_root}/${srcspk}" "${srcspk}" data
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Upstream feature extraction"

    pids=()
    for name in "${srcspk}_train_${num_train}" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}_train_${num_train}" "${trgspk}_dev" "${trgspk}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/${npvc_name}/raw" ] && mkdir -p "${dumpdir}/${name}/${npvc_name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/${npvc_name}/raw/upstream_extraction.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/${npvc_name}/raw"
        CUDA_VISIBLE_DEVICES="" ${train_cmd} JOB=1:${n_jobs} --gpu 0 "${dumpdir}/${name}/${npvc_name}/raw//upstream_extraction.JOB.log" \
            s3prl-vc-extract-upstream \
                --scp "${dumpdir}/${name}/${npvc_name}/raw/wav.JOB.scp" \
                --checkpoint "${npvc_checkpoint}" \
                --config "${npvc_config}" \
                --outdir "${dumpdir}/${name}/${npvc_name}/raw/dump.JOB" \
                --feat_type "${npvc_name}" \
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
            echo "Statistics computation start. See the progress via ${dumpdir}/${name}/${npvc_name}/compute_statistics.log."
            ${train_cmd} "${dumpdir}/${name}/${npvc_name}/compute_statistics.log" \
                compute_statistics.py \
                    --config "${conf}" \
                    --rootdir "${dumpdir}/${name}/${npvc_name}/raw" \
                    --dumpdir "${dumpdir}/${name}/${npvc_name}" \
                    --feat_type "${npvc_name}" \
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
            [ ! -e "${dumpdir}/${name}/${npvc_name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/${npvc_name}/norm_${norm_name}"
            echo "Nomalization start. See the progress via ${dumpdir}/${name}/${npvc_name}/norm_${norm_name}/normalize.*.log."
            ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/${npvc_name}/norm_${norm_name}/normalize.JOB.log" \
                normalize.py \
                    --config "${config_for_feature_extraction}" \
                    --stats "${stats}" \
                    --rootdir "${dumpdir}/${name}/${npvc_name}/raw/dump.JOB" \
                    --dumpdir "${dumpdir}/${name}/${npvc_name}/norm_${norm_name}/dump.JOB" \
                    --feat_type "${npvc_name}" \
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
    expname=${srcspk}_${trgspk}_${num_train}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}_${num_train}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network training"

    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet. Usually parallel VC training can be done with 1 GPU."
        exit 1
    fi

    src_train_dumpdir="${dumpdir}/${srcspk}_train_${num_train}/${npvc_name}/norm_${norm_name}"
    src_dev_dumpdir="${dumpdir}/${srcspk}_dev/${npvc_name}/norm_${norm_name}"
    trg_train_dumpdir="${dumpdir}/${trgspk}_train_${num_train}/${npvc_name}/norm_${norm_name}"
    trg_dev_dumpdir="${dumpdir}/${trgspk}_dev/${npvc_name}/norm_${norm_name}"

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
                --src-train-dumpdir "${src_train_dumpdir}" \
                --src-dev-dumpdir "${src_dev_dumpdir}" \
                --src-feat-type "${npvc_name}" \
                --trg-train-dumpdir "${trg_train_dumpdir}" \
                --trg-dev-dumpdir "${trg_dev_dumpdir}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --trg-feat-type "${npvc_name}" \
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
                --dumpdir "${dumpdir}/${name}/${npvc_name}/norm_${norm_name}/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --src-feat-type "${npvc_name}" \
                --trg-feat-type "${npvc_name}" \
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
            ../cascade/local/evaluate.py \
                --wavdir "${outdir}/${name}" \
                --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --asr
        grep "Mean CER" "${outdir}/${name}/evaluation.log"
    done
fi