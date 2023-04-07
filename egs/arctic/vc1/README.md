# Parallel one-to-one seq2seq VC recipe using the CMU ARCTIC dataset based on the Voice Transformer Network (VTN)

CMU ARCTIC is a dataset containing 1132 parallel utterances (= with the same contents) from several speakers. Here we perform parallel one-to-one VC, which means the model can convert the source speaker's voice to that of the target speaker where both source and target speakers were the same as in the training.

This recipe can be used to (almost) reproduce the results in the [VTN paper](https://arxiv.org/abs/2110.06280). In particular, in the paper, we experimented with mainly two data sizes: 932 utterances (~1 hour) and 80 utterances (~5 mins).

## Train from scratch

With a sufficient amount of data (around 1 hour), VTN can somewhat generate meaningful (but not good) results.

```
./run.sh --stage -1 --stop_stage 3 --norm_name self --tag <experiment_tag_name>
```

- Stage -1: Data and pre-trained model download. First, the CMU ARCTIC dataset is downloaded (by default to `downloads/`). Then, a pre-trained vocoder is also downloaded (to `downloads/pwg_<trgspk>`). We use a Parallel WaveGAN vocoder.
- Stage 0: Data preparation. File lists are generated in `data/<spk>_<set>_<num_train_utterances>` by default. Each file contains space-separated lines with the format `<id> <wave file path>`. These files are used for training and decoding (conversion).
- Stage 1: Feature extraction. The raw (=unnormalized) mel spectrogram for each training utterance is extracted and saved in `dump/<spk>_<set>_<num_train_utterances>/raw`.
- Stage 2: Statistics calculation and normalization. When training from scratch, the statistics of the mel spectrogram used for normalzation are calculated using the training set and saved in `dump/<spk>_train_<num_train_utterances>/stats.h5`. Then, the normalized mel spectrograms are saved in `dump/<spk>_<set>_<num_train_utterances>/norm_<norm_name>`.
- Stage 3: Main training script. By default, `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Modifiable arguments:
- `--norm_name`: if training from scratch, we must pass `self` to this argument. Then, the statistics is calculated based on the training set, and the source and target training sets are normalized w.r.t. the respective statistics.
- `--srcspk`: The VTN paper used `clb` and `bdl` as the source speakers.
- `--trgspk`: The VTN paper used `slt` and `rms` as the source speakers.
- `--tag`: training results will be saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>`.

Then, decoding (conversion) and evaluation can be done by executing the following:

```
./run.sh --stage 4 --stop_stage 5 --norm_name self --tag <experiment_tag_name> --checkpoint <checkpoint_path>
```

- Stage 4: Decoding (conversion). In this stage, the converted mel spectrograms and waveforms are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/out.X`. Specifically, several directories can be found:
  - `att_ws`: the alignment plots.
  - `feats`: converted mel spectrogram files, saved in h5 format.
  - `outs`: visualization of the converted mel spectrograms.
  - `probs`: visualization per-frame predicted stop token probability.
  - `wav`: final generated wav files.
- Stage 5: Objective evaluation. Results are saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>/results/checkpoint-XXsteps/<srcspk>_<set>/evaluation.log` In total we calculated the following metrics:
  - `MCD`: mel cepstrum distortion.
  - `f0RMSE` and `f0CORR`: RMSE and linear correlation coefficient of f0.
  - `DDUR`: absolute duration difference.
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.

## Fine-tune using TTS pre-trained model

Using a pre-trained model can speed up training and improve performance. We provide a pre-trained model which was conducted on LJSpeech. The model is hosted by HuggingFace Hub in: https://huggingface.co/unilight/seq2seq-vc/tree/main/ljspeech/transformer_tts_aept. Alternatively, you can also perform your own pre-training. Please refer to `egs/ljspeech/tts1` for more details

```
./run.sh --stage -1 --stop_stage 3 \
    --conf conf/vtn.tts_pt.v1.yaml \
    --norm_name ljspeech \
    --tag <experiment_tag>
```

- Stage -1: Data and pre-trained model download. The dataset and pre-trained vocoder are downloaded same as above. In addition, the pre-trained model will be downloaded and saved in `downloads/pretrained_models`.
- Stage 0: Data preparation. Same as above.
- Stage 1: Feature extraction. Same as above.
- Stage 2: Statistics calculation and normalization. When using a pre-trained model, the statistics of the pre-trained dataset (LJSpeech) is directly used. The normalized mel spectrograms are still saved in `dump/<spk>_<set>_<num_train_utterances>/norm_<norm_name>`.
- Stage 3: Main training script. Same as above.

Notes:
- The `--pretrained_model_checkpoint` decides which pre-trained model to use. The script assumes that the statistics file is named `stats.h5` and the config file is named `config.yml`, and are both placed in the same directory as the checkpoint.
The name `ljspeech` for the `--norm_name` argument has no actual meaning, and is only used to distinguish features normalized using different statistics. 

## Decoding (conversion) and evaluation

Decoding (conversion) and evaluation can be done by executing the following:

```
./run.sh --stage 4 --stop_stage 5 --norm_name ljspeech --tag <experiment_tag_name> --checkpoint <checkpoint_path>
```

Please refer to the above for explanations.