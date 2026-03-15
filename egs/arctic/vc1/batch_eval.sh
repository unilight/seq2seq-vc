for srcspk in clb bdl; do
    for trgspk in slt rms; do
        ./run.sh --stage 5 --stop_stage 5 --norm_name ljspeech --conf conf/vtn.tts_pt.v1.yaml --srcspk ${srcspk} --trgspk ${trgspk} --tag tts_pt_r1
    done
done
