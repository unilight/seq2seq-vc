
import logging
from fastdtw import fastdtw
import librosa
import numpy as np
import pysptk
import pyworld as pw
import scipy
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from torch._C import ErrorReport

MCEP_DIM=39
MCEP_ALPHA=0.466
MCEP_SHIFT=5
MCEP_FFTL=1024

def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER
    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter
    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x

def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram
    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope
    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence
    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow

def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power
    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2
    Return
    ------
    power : scala,
        Power of a frame
    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power

def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold
    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20
    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`
    """

    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata

def world_extract(x, fs, f0min, f0max):
    # scale from [-1, 1] to [-32768, 32767]
    x = x * np.iinfo(np.int16).max
    
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=MCEP_SHIFT
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    mcep = pysptk.sp2mc(sp, MCEP_DIM, MCEP_ALPHA)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }

def calculate_mcd_f0(x, y, fs, f0min, f0max):
    """
    x and y must be in range [-1, 1]
    """

    # extract ground truth and converted features
    gt_feats = world_extract(x, fs, f0min, f0max)
    cvt_feats = world_extract(y, fs, f0min, f0max)

    # VAD & DTW based on power
    gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
    cvt_mcep_nonsil_pow = extfrm(cvt_feats["mcep"], cvt_feats["npow"])
    _, path = fastdtw(cvt_mcep_nonsil_pow, gt_mcep_nonsil_pow, dist=scipy.spatial.distance.euclidean)
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
        _, path = fastdtw(cvt_mcep_nonsil_f0, gt_mcep_nonsil_f0, dist=scipy.spatial.distance.euclidean)
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

    return mcd, f0rmse, f0corr, ddur
