# Cascade method for foreign accent conversion on the L2-ARCTIC and ARCTIC datasets

There are two stages in the cascade method.
- The first stage is the seq2seq mapping
- The second stage is speaker identity restoration with a non-parallel frame-based model.

![](https://unilight.github.io/Publication-Demos/publications/fac-evaluate/imgs/method.png)

## Preparation and training

```
./run.sh --stage -1 --stop_stage 3 --tag <experiment_tag_name>
```

- Stage -1: Data and pre-trained model download.
    - Make sure L2-ARCTIC is downloaded to somewhere locally and properly set in `run.sh`.
    - The CMU ARCTIC dataset is downloaded (by default to `downloads/`).
    - A pre-trained vocoder is downloaded (to `downloads/pwg_bdl`). We use a Parallel WaveGAN vocoder.
    - The non-parallel frame-based model is downloaded (to `downloads/s3prl-vc-ppg_sxliu`).
    - The pre-trained seq2seq VC model is downloaded (to `downloads/ljspeech_transformer_tts_aept`).
- Stage 0: Data preparation. File lists are generated in `data/<spk>_<set>_<num_train_utterances>` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training and decoding (conversion).
- Stage 1: Feature extraction. The raw (=unnormalized) mel spectrogram for each training utterance is extracted and saved in `dump/<spk>_<set>_<num_train_utterances>/raw`.
- Stage 2: Statistics calculation and normalization. The normalized mel spectrograms are saved in `dump/<spk>_<set>_<num_train_utterances>/norm_<norm_name>`.
- Stage 3: Main training script. By default, `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--tag`: training results will be saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>`.

## Decoding & evaluation

```
./run.sh --stage 4 --stop_stage 5 --norm_name self --tag <experiment_tag_name>
```

- Stage 4: **First stage** decoding (conversion). In this stage, the converted mel spectrograms and waveforms are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/out.X`.
- Stage 5: **First stage** objective evaluation. Results are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/evaluation.log` In total we calculated the following metrics:
  - `MCD`: mel cepstrum distortion.
  - `f0RMSE` and `f0CORR`: RMSE and linear correlation coefficient of f0.
  - `DDUR`: absolute duration difference.
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.
- Stage 6: **Second stage** decoding (conversion). In this stage, the converted mel spectrograms and waveforms are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/stage2_<npvc_name>_checkpoint-XXsteps/<srcspk>_<set>/out.X`.
- Stage 7: **Second stage** objective evaluation. Results are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/stage2_<npvc_name>_checkpoint-XXsteps/<srcspk>_<set>/evaluation.log` In total we calculated the following metrics:
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.
