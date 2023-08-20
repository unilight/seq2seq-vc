#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
import torchaudio
from tqdm import tqdm
import yaml

from seq2seq_vc.utils import find_files
from seq2seq_vc.utils.types import str2bool
from seq2seq_vc.evaluate.dtw_based import calculate_mcd_f0
from seq2seq_vc.evaluate.asr import load_asr_model, transcribe, calculate_measures

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions",  "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename] # get rid of the first character "E"
        
        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(groundtruth, transcription)

        ers[basename] = [c_result["cer"] * 100.0, w_result["wer"] * 100.0, norm_transcription, norm_groundtruth]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]
  
    # calculate over whole set
    def er(r):
        return float(r["substitutions"] + r["deletions"] + r["insertions"]) \
            / float(r["substitutions"] + r["deletions"] + r["hits"]) * 100.0

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def _calculate_mcd_f0(file_list, gt_root, segments, trgspk, f0min, f0max, results, gv=False):
    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, basename + ".wav")

        # read both converted and ground truth wav
        cvt_wav, cvt_fs = librosa.load(cvt_wav_path, sr=None)
        if segments is not None:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None,
                                         offset=segments[basename]["offset"],
                                         duration=segments[basename]["duration"]
                                         )
        else:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None)
        if cvt_fs != gt_fs:
            cvt_wav = torchaudio.transforms.Resample(cvt_fs, gt_fs)(torch.from_numpy(cvt_wav)).numpy()

        # calculate MCD, F0RMSE, F0CORR and DDUR
        res = calculate_mcd_f0(cvt_wav, gt_wav, gt_fs, f0min, f0max, calculate_gv=gv)

        results.append([basename, res])

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--wavdir", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--data_root", type=str, default="./data", help="directory of data")
    parser.add_argument("--segments", type=str, default=None, help="segments file")
    parser.add_argument("--f0_path", required=True, type=str, help="yaml file storing f0 ranges")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    parser.add_argument("--gv", default=False, type=str2bool, help="calculate GV or not")
    return parser


def main():
    args = get_parser().parse_args()

    trgspk = args.trgspk
    gt_root = os.path.join(args.data_root, "wav")
    transcription_path = os.path.join(args.data_root, "etc", "arctic.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # load f0min and f0 max
    with open(args.f0_path, 'r') as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    f0min = f0_all[trgspk]["f0min"]
    f0max = f0_all[trgspk]["f0max"]

    # load ground truth transcriptions
    with open(transcription_path, "r") as f:
        lines = f.read().splitlines()
    groundtruths = {line.split(" ")[1]: " ".join(line.split(" ")[2:-1]).replace('"', '') for line in lines}

    # load segments if provided
    if args.segments is not None:
        with open(args.segments, "r") as f:
            lines = f.read().splitlines()
        segments = {}
        for line in lines:
            _id, _, start, end = line.split(" ")
            segments[_id] = {
                "offset": float(start),
                "duration": float(end) - float(start)
            }
    else:
        segments = None

    # find converted files
    converted_files = sorted(find_files(args.wavdir, query="*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    ##############################

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(asr_model, device, converted_files, groundtruths)
    
    ##############################

    print("Calculating MCD and f0-related scores...")
    # Get and divide list
    file_lists = np.array_split(converted_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        results = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(
                target=_calculate_mcd_f0,
                args=(f, gt_root, segments, trgspk, f0min, f0max, results, args.gv),
            )
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        sorted_results = sorted(results, key=lambda x:x[0])
        results = []
        for result in sorted_results:
            d = {k: v for k, v in result[1].items()}
            d["basename"] = result[0]
            d["CER"] = ers[result[0]][0]
            d["WER"] = ers[result[0]][1]
            d["GT_TRANSCRIPTION"] = ers[result[0]][2]
            d["CV_TRANSCRIPTION"] = ers[result[0]][3]
            results.append(d)
        
    # utterance wise result
    for result in results:
        print(
            "{} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} {:.1f} \t{} | {}".format(
                result["basename"],
                result["MCD"],
                result["F0RMSE"],
                result["F0CORR"],
                result["DDUR"],
                result["CER"],
                result["WER"],
                result["GT_TRANSCRIPTION"],
                result["CV_TRANSCRIPTION"],
            )
        )

    # average result
    mMCD = np.mean(np.array([result["MCD"] for result in results]))
    mf0RMSE = np.mean(np.array([result["F0RMSE"] for result in results]))
    mf0CORR = np.mean(np.array([result["F0CORR"] for result in results]))
    mDDUR = np.mean(np.array([result["DDUR"] for result in results]))
    mCER = cer 
    mWER = wer

    if not args.gv:
        print(
            "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f}".format(
                mMCD, mf0RMSE, mf0CORR, mDDUR, mCER, mWER
            )
        )
    else:
        mGV = np.mean(np.array([result["GV"] for result in results]))
        print(
            "Mean MCD, GV, f0RMSE, f0CORR, DDUR, CER, WER: {:.2f} {:.3f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f}".format(
                mMCD, mGV, mf0RMSE, mf0CORR, mDDUR, mCER, mWER
            )
        )


if __name__ == "__main__":
    main()