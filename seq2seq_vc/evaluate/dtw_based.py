import logging
from fastdtw import fastdtw
import librosa
import numpy as np
import scipy
from scipy.io import wavfile

from seq2seq_vc.utils.signal import world_extract, extfrm


def calculate_mcd_f0(x, y, fs, f0min, f0max, calculate_gv=False):
    """
    x and y must be in range [-1, 1]
    """

    # extract ground truth and converted features
    gt_feats = world_extract(x, fs, f0min, f0max)
    cvt_feats = world_extract(y, fs, f0min, f0max)

    # VAD & DTW based on power
    gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
    cvt_mcep_nonsil_pow = extfrm(cvt_feats["mcep"], cvt_feats["npow"])
    _, path = fastdtw(
        cvt_mcep_nonsil_pow, gt_mcep_nonsil_pow, dist=scipy.spatial.distance.euclidean
    )
    twf_pow = np.array(path).T

    # MCD using power-based DTW
    cvt_mcep_dtw_pow = cvt_mcep_nonsil_pow[twf_pow[0]]
    gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
    diff2sum = np.sum((cvt_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    # VAD & DTW based on f0
    gt_nonsil_f0_idx = np.where(gt_feats["f0"] > 0)[0]
    cvt_nonsil_f0_idx = np.where(cvt_feats["f0"] > 0)[0]
    try:
        gt_mcep_nonsil_f0 = gt_feats["mcep"][gt_nonsil_f0_idx]
        cvt_mcep_nonsil_f0 = cvt_feats["mcep"][cvt_nonsil_f0_idx]
        _, path = fastdtw(
            cvt_mcep_nonsil_f0, gt_mcep_nonsil_f0, dist=scipy.spatial.distance.euclidean
        )
        twf_f0 = np.array(path).T

        # f0RMSE, f0CORR using f0-based DTW
        cvt_f0_dtw = cvt_feats["f0"][cvt_nonsil_f0_idx][twf_f0[0]]
        gt_f0_dtw = gt_feats["f0"][gt_nonsil_f0_idx][twf_f0[1]]
        f0rmse = np.sqrt(np.mean((cvt_f0_dtw - gt_f0_dtw) ** 2))
        f0corr = scipy.stats.pearsonr(cvt_f0_dtw, gt_f0_dtw)[0]
    except ValueError:
        logging.warning(
            "No nonzero f0 is found. Skip f0rmse f0corr computation and set them to NaN. "
            "This might due to unconverge training. Please tune the training time and hypers."
        )
        f0rmse = np.nan
        f0corr = np.nan

    # DDUR
    # energy-based VAD with librosa
    x_trim, _ = librosa.effects.trim(y=x)
    y_trim, _ = librosa.effects.trim(y=y)
    ddur = float(abs(len(x_trim) - len(y_trim)) / fs)

    ret = {"MCD": mcd, "F0RMSE": f0rmse, "F0CORR": f0corr, "DDUR": ddur}

    # GV
    if calculate_gv:
        # (1) take var along time axis (2) take log (3) take abs (4) take mean
        gv = np.mean(np.abs(np.log(np.var(cvt_mcep_nonsil_pow, axis=0))))
        ret["GV"] = gv

    return ret
