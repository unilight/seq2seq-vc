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
arctic_db_root=../../arctic/vc1/downloads       # default saved here
db_root=/data/group1/z44476r/Corpora/l2-arctic  # PLEASE CHANGE THIS
dumpdir=dump                                    # directory to dump full features
srcspk=TXHC
trgspk=bdl
num_train=1032
stats_ext=h5
norm_name=ljspeech                              # used to specify normalized data.
feat_type=mel

# pretrained model related
pretrained_model_checkpoint=downloads/ljspeech_transformer_tts_aept/checkpoint-50000steps.pkl
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
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/checkpoint-400000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "pwg_TXHC/stats.h5"

    # download npvc model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "s3prl-vc-ppg_sxliu/stats.h5"

    # download pretrained seq2seq model
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_transformer_tts_aept/checkpoint-50000steps.pkl"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_transformer_tts_aept/config.yml"
    utils/hf_download.py --repo_id "unilight/accent-conversion-2023" --outdir "downloads" --filename "ljspeech_transformer_tts_aept/stats.h5"
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
    echo "Stage 1: Synthetic target generation"

    npvc_pretrained_model_dir=$(dirname ${npvc_checkpoint})
    npvc_stats="${npvc_pretrained_model_dir}/stats.h5"
    echo "NPVC pretrained model checkpoint: ${npvc_checkpoint}"

    for name in "dev" "eval" "train_${num_train}"; do
        new_name=${trgspk}2${srcspk}_${npvc_name}_${name}
        [ ! -e "data/${new_name}/wav" ] && mkdir -p "data/${new_name}/wav"
        echo "Decoding start. See the progress via data/${new_name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "data/${new_name}/decode.log" \
            s3prl-vc-decode \
                --scp "data/${trgspk}_${name}/wav.scp" \
                --checkpoint "${npvc_checkpoint}" \
                --trg-stats "${npvc_stats}" \
                --outdir "data/${new_name}" \
                --verbose "${verbose}"

        echo "Evaluation start. See the progress via data/${new_name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "data/${new_name}/evaluation.log" \
            ../cascade/local/evaluate.py \
                --wavdir "data/${new_name}" \
                --data_root "${arctic_db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml" \
                --asr

        echo "Making scp. See results: data/${new_name}/wav.scp"
        [ -e "data/${new_name}/wav.scp" ] && rm "data/${new_name}/wav.scp"
        find "$(realpath data/${new_name})" -name "*.wav" -follow | sort | while read -r filename; do
            id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
            echo "${id} ${filename}" >> "data/${new_name}/wav.scp"
        done

        cp "data/${trgspk}_${name}/segments" "data/${new_name}"

        # copy generated feats to dump/<name>/raw_from_vc_model
        [ ! -e "${dumpdir}/${new_name}/raw_from_vc_model" ] && mkdir -p "${dumpdir}/${new_name}/raw_from_vc_model"
        cp data/${new_name}/mel/*.h5 "${dumpdir}/${new_name}/raw_from_vc_model/"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Feature extraction"

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # extract raw features
    pids=()
    for name in "${srcspk}_train_${num_train}" "${srcspk}_dev" "${srcspk}_eval" "${trgspk}2${srcspk}_${npvc_name}_train_${num_train}" "${trgspk}2${srcspk}_${npvc_name}_dev" "${trgspk}2${srcspk}_${npvc_name}_eval"; do
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

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Statistics computation (optional) and normalization"

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
    for spk in ${srcspk} ${trgspk}2${srcspk}_${npvc_name}; do
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

    # normalize feats from VC model directly
    for name in "${trgspk}2${srcspk}_${npvc_name}_train_${num_train}" "${trgspk}2${srcspk}_${npvc_name}_dev" "${trgspk}2${srcspk}_${npvc_name}_eval"; do
    (
        [ ! -e "${dumpdir}/${name}/norm_${norm_name}_from_vc_model" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}_from_vc_model"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}_from_vc_model/normalize.*.log."
        ${train_cmd} "${dumpdir}/${name}/norm_${norm_name}_from_vc_model/normalize.log" \
            normalize.py \
                --config "${config_for_feature_extraction}" \
                --stats "${stats}" \
                --rootdir "${dumpdir}/${name}/raw_from_vc_model" \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}_from_vc_model" \
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
fi


if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}2${srcspk}_${npvc_name}_${num_train}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}2${srcspk}_${npvc_name}_${num_train}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network training"

    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet. Usually parallel VC training can be done with 1 GPU."
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
                --trg-train-dumpdir "${dumpdir}/${trgspk}2${srcspk}_${npvc_name}_train_${num_train}/norm_${norm_name}" \
                --trg-dev-dumpdir "${dumpdir}/${trgspk}2${srcspk}_${npvc_name}_dev/norm_${norm_name}" \
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

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Network decoding"
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

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation"

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