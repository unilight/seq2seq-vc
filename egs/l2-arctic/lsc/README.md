# Latent space conversion (LSC) method for foreign accent conversion on the L2-ARCTIC and ARCTIC datasets

There is only one stage during inference in the LSC method, which is the seq2seq mapping.  
However, before training, a non-parallel frame-based model is used to extract the latent features from the source and target training sets for the seq2seq model training.

![](https://unilight.github.io/Publication-Demos/publications/fac-evaluate/imgs/method.png)

## Preparation

```
./run.sh --stage -1 --stop_stage 2
```

- Stage -1: Data and pre-trained model download.
    - Make sure L2-ARCTIC is downloaded to somewhere locally and properly set in `run.sh`.
    - The CMU ARCTIC dataset is downloaded (by default to `downloads/`).
    - A pre-trained vocoder is downloaded (to `downloads/pwg_TXHC`). We use a Parallel WaveGAN vocoder.
    - The non-parallel frame-based models is downloaded (to `downloads/s3prl-vc-ppg_sxliu` (for latent feature extraction) and `downloads/ppg_sxliu_decoder_THXC` (for latent feature to mel transformation)).
    - **The pre-trained seq2seq VC model is downloaded (to `downloads/ljspeech_text_to_ppg_sxliu_aept`). Note that this pre-trained model is latent-to-latent, not mel-to-mel.**
- Stage 0: Data preparation. File lists are generated in `data/<spk>_<set>_<num_train_utterances>` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training and decoding (conversion).
- Stage 1: **Latent feature extraction**. The non-parallel frame-based model is used to extract the latent features from the source and target training sets for the seq2seq model training. The extracted features are saved in `dump/<spk>_<set>_<num_train_utterances>/<npvc_name>/raw`.
- Stage 2: Statistics calculation and normalization. The normalized latent features are saved in `dump/<spk>_<set>_<num_train_utterances>/<npvc_name>/norm_<norm_name>`.

**IMPORTANT to-do before training!!**

Please update the following lines in `downloads/ppg_sxliu_decoder_THXC/config.yml` with the absolute paths of `downloads/pwg_TXHC/`:

```
vocoder:
  checkpoint: /data/group1/z44476r/Experiments/seq2seq-vc/egs/l2-arctic/lsc/downloads/pwg_TXHC/checkpoint-400000steps.pkl # Please change this line
  config: /data/group1/z44476r/Experiments/seq2seq-vc/egs/l2-arctic/lsc/downloads/pwg_TXHC/config.yml                     # Please change this line
  stats: /data/group1/z44476r/Experiments/seq2seq-vc/egs/l2-arctic/lsc/downloads/pwg_TXHC/stats.h5                        # Please change this line
```

## Training

./run.sh --stage 3 --stop_stage 3 --tag <experiment_tag_name>

- Stage 3: Main training script. By default, `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--tag`: training results will be saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>`.

## Decoding & evaluation

```
./run.sh --stage 4 --stop_stage 5 --tag <experiment_tag_name>
```

- Stage 4: decoding (conversion). In this stage, the converted waveforms are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/out.X`.
- Stage 5: objective evaluation. Results are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/evaluation.log` In total we calculated the following metrics:
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.
