# Parallel one-to-one seq2seq VC recipe using the CMU ARCTIC dataset

We provide two recipes:

- `vc1`: autoregressive (AR) seq2seq modeling, reproducing [1]
- `vc2`: non-autoregressive (non-AR) seq2seq modeling, reproducing [2, 3]
- `vc_urhythmic`: seq2seq VC based on Urhythmic [4]. Ported from https://github.com/bshall/urhythmic.

[1] W.-C. Huang, T. Hayashi, Y.-C. Wu, H. Kameoka, and T. Toda, “Voice Transformer Network: Sequence-to-Sequence Voice Conversion Using Transformer with Text-to-Speech Pretraining,” in Proc. Interspeech, 2020, pp. 4676–4680.  
[2] T. Hayashi, W.-C. Huang, K. Kobayashi, and T. Toda, “Non- Autoregressive Sequence-To-Sequence Voice Conversion,” in Proc. ICASSP, 2021, pp. 7068–7072.  
[3] W.-C. Huang, K. Kobayashi, and T. Toda, “AAS-VC: On the Generalization Ability of Automatic Alignment Search based Non-autoregressive Sequence-to-sequence Voice Conversion,”  rejected by ICASSP2024.  
[4] B. van Niekerk, M. -A. Carbonneau and H. Kamper, "Rhythm Modeling for Voice Conversion," in IEEE Signal Processing Letters, vol. 30, pp. 1297-1301, 2023, doi: 10.1109/LSP.2023.3313515.