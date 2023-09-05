# Synthetic target generation (STG) method for foreign accent conversion on the L2-ARCTIC and ARCTIC datasets

There is only one stage during inference in the STG method, which is the seq2seq mapping.  
However, before training, a non-parallel frame-based model is used to synthesize the training target for the seq2seq model training.

![](https://unilight.github.io/Publication-Demos/publications/fac-evaluate/imgs/method.png)

## Preparation and training

```
./run.sh --stage -1 --stop_stage 4 --tag <experiment_tag_name>
```

- Stage -1: Data and pre-trained model download.
    - Make sure L2-ARCTIC is downloaded to somewhere locally and properly set in `run.sh`.
    - The CMU ARCTIC dataset is downloaded (by default to `downloads/`).
    - A pre-trained vocoder is downloaded (to `downloads/pwg_TXHC`). We use a Parallel WaveGAN vocoder.
    - The non-parallel frame-based model is downloaded (to `downloads/s3prl-vc-ppg_sxliu`).
    - The pre-trained seq2seq VC model is downloaded (to `downloads/ljspeech_transformer_tts_aept`).
- Stage 0: Data preparation. File lists are generated in `data/<spk>_<set>_<num_train_utterances>` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training and decoding (conversion).
- Stage 1: **Synthetic target generation**. The non-parallel frame-based model is used to generate the training target for the seq2seq model training. The generated waveform samples are saved in `data/<trgspk>2<srcspk>_<npvc_name>_<set>`.
- Stage 2: Feature extraction. The raw (=unnormalized) mel spectrogram for each training utterance is extracted and saved in either `dump/<srcspk>_<set>_<num_train_utterances>/raw` or `dump/<trgspk>2<srcspk>_<npvc_name>_<set>_<num_train_utterances>`.
- Stage 3: Statistics calculation and normalization. The normalized mel spectrograms are saved in either `dump/<srcspk>_<set>_<num_train_utterances>/raw` or `dump/<trgspk>2<srcspk>_<npvc_name>_<set>_<num_train_utterances>`.
- Stage 4: Main training script. By default, `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--tag`: training results will be saved in `exp/<srcspk>_<trgspk>2<srcspk>_<npvc_name>_<num_train_utterances>_<tag>`.

## Decoding & evaluation

```
./run.sh --stage 5 --stop_stage 6 --norm_name self --tag <experiment_tag_name>
```

- Stage 5: decoding (conversion). In this stage, the converted mel spectrograms and waveforms are saved in `exp/<srcspk>_<trgspk>2<srcspk>_<npvc_name>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/out.X`.
- Stage 6: objective evaluation. Results are saved in `exp/<srcspk>_<trgspk>2<srcspk>_<npvc_name>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/evaluation.log` In total we calculated the following metrics:
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.
