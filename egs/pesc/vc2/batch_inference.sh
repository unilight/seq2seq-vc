#!/usr/bin/env bash

set -euo pipefail

#for spk in 002 003 008 009 010 011; do
#for spk in 011; do
#for spk in 001 002 003 004 005 006  007 008 009 010 011; do
for spk in 012 013; do
    ./run.sh --stage 4 --stop_stage 5 \
        --srcspk EL_PS_MALE${spk} \
        --trgspk SP_PS_MALE${spk} \
        --checkpoint exp/EL_PS_MALE${spk}_SP_PS_MALE${spk}_aas_vc.melmelmel.v1/checkpoint-100000steps.pkl
done
