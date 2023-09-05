# seq2seq-vc: sequence-to-sequence voice conversion toolkit

Paper (INTERSPEECH2020) [![arXiv](https://img.shields.io/badge/arXiv-1912.06813-b31b1b.svg)](https://arxiv.org/abs/1912.06813)  
Paper (IEEE/ACM TASLP)  [![arXiv](https://img.shields.io/badge/arXiv-2008.03088-b31b1b.svg)](https://arxiv.org/abs/2008.03088)  
Original codebase on ESPNet [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/espnet/espnet/tree/master/egs/arctic/vc1)  

## News

- **2023.09.: We released a recipe for foreign accent conversion! Please refer to `egs/l2-arctic` for details.**

## Introduction and motivation

Sequence-to-sequence (seq2seq) modeling is especially attractive to voice conversion (VC) owing to their ability to convert prosody. In particular, this repository aim to reproduce the results of the following papers/models.

### Voice Transformer Network (VTN) ([paper](https://arxiv.org/abs/1912.06813))
This is the first paper that applies the Transformer model to VC. In addition to the model architecture itself, the true novelty of this paper is actually a pre-training technique based on text-to-speech (TTS). This repository provides recipes for (1) TTS pre-training and (2) fine-tuning on a VC dataset. That is to say, TTS is also available in this repository.

Originally I open-sourced the code on [ESPNet](https://github.com/espnet/espnet), but as it grows bigger and bigger, it becomes harder to conduct scientific research on ESPNet. Therefore, this repository aims to isolate the seq2seq VC part from ESPNet to become an independently-maintained toolkit (hopefully).


## Instsallation 

### Editable installation with virtualenv 

```
git clone https://github.com/unilight/s3prl-vc.git
cd s3prl-vc/tools
make
```

## Complete training, decoding and benchmarking

Same as many speech processing based repositories ([ESPNet](https://github.com/espnet/espnet), [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), etc.), we formulate our recipes in kaldi-style. They can be found in the `egs` folder. Please check the detailed usage in each recipe.

### Reproducing VTN experiments

1. The TTS pre-training is conducted on LJSpeech. Please refer to the readme file in `egs/ljspeech/tts1`.
2. Afterwards, the VC fine-tuning is conducted on CMU ARCTIC. Please refer to the readme file in `egs/arctic/vc1`.

## Citation

```
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
