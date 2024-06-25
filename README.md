# seq2seq-vc: sequence-to-sequence voice conversion toolkit

![visitors](https://visitor-badge.laobi.icu/badge?page_id=unilight.seq2seq-vc)

_**(NEW)**_ Paper (Submitted to ICASSP 2024) [![arXiv](https://img.shields.io/badge/arXiv-2309.07598-b31b1b.svg)](https://arxiv.org/abs/2309.07598)  
_**(NEW)**_ Paper (APSIPA ASC 2023) [![arXiv](https://img.shields.io/badge/arXiv-2309.02133-b31b1b.svg)](https://arxiv.org/abs/2309.02133)  
Paper (INTERSPEECH 2020) [![arXiv](https://img.shields.io/badge/arXiv-1912.06813-b31b1b.svg)](https://arxiv.org/abs/1912.06813)  
Paper (IEEE/ACM TASLP)  [![arXiv](https://img.shields.io/badge/arXiv-2008.03088-b31b1b.svg)](https://arxiv.org/abs/2008.03088)  
Original codebase on ESPNet [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/espnet/espnet/tree/master/egs/arctic/vc1)  

## News

- **2024.03**: I ported the [Urhythmic](https://github.com/bshall/urhythmic) method to this repo. All credits go to the original author. Please check `egs/arctic/vc_urhythmic`.
- **2023.10**: We released a recipe for running AAS-VC on [Hi-Fi-CAPTAIN](https://ast-astrec.nict.go.jp/en/release/hi-fi-captain/)! Please refer to `egs/hificaptain_jp/vc2` for details.
- **2023.09**: We released a recipe for non-autoregressive seq2seq VC! Please refer to `egs/arctic/vc2` for details.
- **2023.09**: We released a recipe for foreign accent conversion (FAC)! Please refer to `egs/l2-arctic` for details.

## Introduction and motivation

Sequence-to-sequence (seq2seq) modeling is especially attractive to voice conversion (VC) owing to their ability to convert prosody. In particular, this repository aim to reproduce the results of the following papers/models.

### Automatic alignment search (AAS) based non-autoregressive seq2seq VC ([paper](https://arxiv.org/abs/2309.07598))
A non-autoregressive seq2seq VC method that can be directly trained on a parallel dataset without any pre-trained models.

### Ground-Truth-Free Foreign Accent Conversion (FAC) ([paper](https://arxiv.org/abs/2309.02133))
We compared three methods for ground-truth-free FAC. All methods utilize seq2seq modeling and non-parallel frame-based VC models. We provide pre-trained vocoder and non-parallel frame-based VC models.

### Voice Transformer Network (VTN) ([paper](https://arxiv.org/abs/1912.06813))
This is the first paper that applies the Transformer model to VC. In addition to the model architecture itself, the true novelty of this paper is actually a pre-training technique based on text-to-speech (TTS). This repository provides recipes for (1) TTS pre-training and (2) fine-tuning on a VC dataset. That is to say, TTS is also available in this repository.

Originally I open-sourced the code on [ESPNet](https://github.com/espnet/espnet), but as it grows bigger and bigger, it becomes harder to conduct scientific research on ESPNet. Therefore, this repository aims to isolate the seq2seq VC part from ESPNet to become an independently-maintained toolkit (hopefully).


## Instsallation 

### Editable installation with virtualenv 

```
git clone https://github.com/unilight/seq2seq-vc.git
cd seq2seq-vc/tools
make
```

## Complete training, decoding and benchmarking

Same as many speech processing based repositories ([ESPNet](https://github.com/espnet/espnet), [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), etc.), we formulate our recipes in kaldi-style. They can be found in the `egs` folder. Please check the detailed usage in each recipe.

### Reproducing AAS-VC experiments
The experiments in the AAS-VC paper are conducted using the CMU ARCTIC dataset. Please refer to the readme file in `egs/arctic/vc2`.

### Reproducing FAC experiments

The CMU ARCTIC and L2-ARCTIC datasets are used. Please refer to the readme file in `egs/l2-arctic`.

### Reproducing VTN experiments

1. The TTS pre-training is conducted on LJSpeech. Please refer to the readme file in `egs/ljspeech/tts1`.
2. Afterwards, the VC fine-tuning is conducted on CMU ARCTIC. Please refer to the readme file in `egs/arctic/vc1`.

## Citation

```
@INPROCEEDINGS{fac-evaluate,
  author={Wen-Chin Huang and Tomoki Toda},
  booktitle={Proc. APSIPA ASC},
  title={{Evaluating Methods for Ground-Truth-Free Foreign Accent Conversion}},
  year={2023},
}
@inproceedings{huang20i_interspeech,
  author={Wen-Chin Huang and Tomoki Hayashi and Yi-Chiao Wu and Hirokazu Kameoka and Tomoki Toda},
  title={{Voice Transformer Network: Sequence-to-Sequence Voice Conversion Using Transformer with Text-to-Speech Pretraining}},
  year=2020,
  booktitle={Proc. Interspeech},
  pages={4676--4680},
}
@ARTICLE{vtn_journal,
  author={Huang, Wen-Chin and Hayashi, Tomoki and Wu, Yi-Chiao and Kameoka, Hirokazu and Toda, Tomoki},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Pretraining Techniques for Sequence-to-Sequence Voice Conversion}, 
  year={2021},
  volume={29},
  pages={745-755},
}
```

## Acknowledgements

This repo is greatly inspired by the following repos. Or I should say, many code snippets are directly taken from part of the following repos.

- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)

## Author

Wen-Chin Huang  
Toda Labotorary, Nagoya University  
E-mail: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp
