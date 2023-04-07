import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_attention(array, figname, figsize=(6, 4), dpi=150, origin="upper"):
    shape = array.shape
    # for transformer attention weights,
    # whose shape is (#leyers, #heads, out_length, in_length)
    plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
    for idx1, xs in enumerate(array):
        for idx2, x in enumerate(xs, 1):
            plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
            plt.imshow(x, aspect="auto")
            plt.xlabel("Input")
            plt.ylabel("Output")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()


def plot_generated_and_ref_2d(
    array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
):
    if ref is None:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(array.T, aspect="auto", origin=origin)
        plt.xlabel("Frame")
        plt.ylabel("Frequency")
    else:
        plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
        plt.subplot(1, 2, 1)
        plt.imshow(array.T, aspect="auto", origin=origin)
        plt.xlabel("Frame")
        plt.ylabel("Frequency")
        plt.subplot(1, 2, 2)
        plt.imshow(ref.T, aspect="auto", origin=origin)
        plt.xlabel("Frame")
        plt.ylabel("Frequency")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()


def plot_1d(array, figname, figsize=(6, 4), dpi=150, origin="upper"):
    # for eos probability
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(array)
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.ylim([0, 1])

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()
