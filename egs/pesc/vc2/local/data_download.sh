#!/usr/bin/env bash
set -e

# Copyright 2024 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

# download dataset
cwd=`pwd`
if [ ! -e ${db}/pesc.done ]; then
    mkdir -p ${db}
    cd ${db}
    # wget https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_F.zip
    # wget https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_M.zip
    # unzip hfc_ja-JP_F.zip
    # unzip hfc_ja-JP_M.zip
    # rm hfc_ja-JP_F.zip
    # rm hfc_ja-JP_M.zip
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/pesc.done
else
    echo "Already exists. Skip download."
fi
