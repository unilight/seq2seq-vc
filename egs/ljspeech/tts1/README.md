# TTS & TTS autoencoder pre-training recipe using the LJSpeech dataset based on the Transformer network

LJSpeech is a commonly used 22.05 kHz TTS dataset containing 24 hours of English speech from a single female speaker.

This recipe can be used to perform the TTS autoencoder pretraining described in the [VTN paper](https://arxiv.org/abs/2110.06280). In particular, a TTS model needs to be trained first. Then, the decoder is fixed and the mel spectrogram encoder is trained in an autoencoder fashion. Note that since the VC dataset, CMu ARCTIC, has a 16kHz sampling rate, we also downsample LJSpeech from 22.05 kHz to 16kHz.

## TTS model training

```
./run.sh --stage -1 --stop_stage 3 --tag <experiment_tag_name>
```

- Stage -1: Data and pre-trained model download. The LJSpeech dataset is downloaded (by default to `downloads/`).
- Stage 0: Data preparation. File lists and text files are generated in `data/<set>` by default. Each file list contains space-separated lines with the format `<id> <wave file path>`. Each text file contains space-separated lines with the format `<id> <text>`. These files are used for training and decoding.
- Stage 1: Feature extraction, statistics calculation and normalization. First, the raw (=unnormalized) mel spectrogram for each training utterance is extracted and saved in `dump/<set>/raw`. Then, the statistics of the mel spectrogram used for normalization are calculated using the training set and saved in `dump/train_no_dev/stats.h5`. Then, the normalized mel spectrograms are saved in `dump/<set>/norm`.
- Stage 2: Token generation. A list of tokens is generated using the training texts, using the text preprocessing arguments. The tokenization log and the generated token list are saved in `dump/token_list/<token_type>_<text_cleaner>_<g2p_tool>`.
- Stage 3: Main training script. By default, `exp/<token_type>_<cleaner>_<tag>` is used to save the training log, saved checkpoints and intermediate samples for debugging (saved in `predictions/`).

Important modifiable arguments:
- `--token_type`. There are main two choices: character (`char`) and phoneme (`phn`). Usually using phonemes is better, but since LJSpeech is rather large, using characters can also yield good results.

### Decoding

Decoding is not necessary, but you can use this step to investigate whether the TTS model is well-trained. Note that we did not optimize this process using multi-processing, so this could take more than 20 minutes.

```
./run.sh --stage 4 --stop_stage 5 --tag <experiment_tag_name> --checkpoint <checkpoint_path>
```

- Stage 4: Decoding. In this stage, the generated mel spectrograms and waveforms are saved in `exp/<token_type>_<cleaner>_<tag>/results/checkpoint-XXsteps/<set>`. Specifically, several directories can be found:
  - `att_ws`: the alignment plots.
  - `outs`: visualization of the generated mel spectrograms.
  - `probs`: visualization per-frame predicted stop token probability.
  - `wav`: final generated wav files. We use the Griffin-Lim (GL) algorithm here.
- Stage 5: Objective evaluation. Results are saved in `exp/<token_type>_<cleaner>_<tag>/results/checkpoint-XXsteps/<set>/evaluation.log` In total we calculated the following metrics. Note that due to the use of GL, the MCD will be rather high. It would be fine to just look at CER and WER.
  - `MCD`: mel cepstrum distortion.
  - `f0RMSE` and `f0CORR`: RMSE and linear correlation coefficient of f0.
  - `DDUR`: absolute duration difference.
  - `CER` and `WER`: character/word error rates from a pre-trained ASR model.

## TTS autoencoder pre-training

As stated above, after we train the TTS model, we can perform TTS autoencoder pre-training. Specifically, the decoder is fixed and the mel spectrogram encoder is trained in an autoencoder fashion.

```
./run.sh --stage 6 --stop_stage 6 --tts_aept_exptag <tag_for_aept> --tts_aept_checkpoint <tts_checkpoint>
```

The config file is specified with the argument `--tts_aept_config` and is by default set to `conf/tts_aept.v1.yaml`. Training results can be found in `exp/tts_aept_${tts_aept_origin_expdir}_${tts_aept_checkpoint_name}`.