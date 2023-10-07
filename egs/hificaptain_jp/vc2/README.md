# Parallel one-to-one non-autoregreseive (non-AR) seq2seq VC recipe using the Hi-Fi-CAPTAIN dataset

2023.10.

This recipe runs AAS-VC (a non-AR seq2seq VC model) training on the [Hi-Fi-CAPTAIN](https://ast-astrec.nict.go.jp/en/release/hi-fi-captain/) dataset, a large-scale (over 20 hours) parallel English & Japanese 48kHz speech dataset. **Currently we only support Japanese male to Japanese female conversion**. Please let me know if you want to use other datasets.

This README describe the basic usage. For more details regarding AAS-VC, please refer to `arctic/vc2`.

### What if OOM happens during training?

The default AAS-VC recipe runs with a batch size of 16. Since Hi-Fi-CAPTAIN is 48kHz, the sequence lengths become extremely long. On a Tesla V100 with 32GB memory, I had to use batch size = 2 with gradient accumulate steps = 8. It took 40 hours to reach 50k steps. Please try different combinations according to your GPU spec.

### Preparation

First, download materials (dataset, pre-trained vocoder and AAS-VC model), extract the features, calculate the statistics and normalize all the features.

```
./run.sh --stage -1 --stop_stage 2 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ
```

The dataset is quite large (in total 20GB) and might take a while to download.

It is always recommended to explicitly set `--src_feat`, `--trg_feat` and `--dp_feat`.

### Training

```
./run.sh --stage 3 --stop_stage 3 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name>
```

Training results will be saved in `exp/<srcspk>_<trgspk>_<tag>`.

### Decoding and evaluation

Then, decoding (conversion) and evaluation can be done by executing the following:

```
./run.sh --stage 4 --stop_stage 5 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name>
```

### Decoding with already trained model

To quickly obtain results, a already trained model is automatically downloaded to `exp/male_female_aas_vc_mel_pretrained` in stage -1. To decode, simply use the correct tag:

```
./run.sh --stage 4 --stop_stage 5 --tag aas_vc_mel_pretrained
```