# Recipes for foreign accent conversion on the L2-ARCTIC and ARCTIC datasets

L2-ARCTIC is a speech corpus of non-native English speakers uttering the same prompts as those in the ARCTIC dataset. Here we provide recipes to the three methods described in our paper, "Evaluating Methods for Ground-Truth-Free Foreign Accent Conversion".

- Method 1: cascade -> `cascade`
- Method 2: synthetic target generation (STG) -> `stg`
- Method 3: latent space conversion (LSC) -> `lsc`

![](https://unilight.github.io/Publication-Demos/publications/fac-evaluate/imgs/method.png)

## Datasets

To get started, you must first obtain the L2-ARCTIC dataset and the ARCTIC datasets. The ARCTIC download script is included in each recipe. The L2-ARCTIC dataset needs to be obtained from their [official website](https://psi.engr.tamu.edu/l2-arctic-corpus/) by following their instruction. The datasets can be put anywhere in your local server, as long as the `db_root` and `arctic_db_root` variables are properly set in the `run.sh` files in each recipe.

The default source speaker is THXC (Chinese male) from the L2-ARCTIC dataset and the target speaker bdl (English male) from the ARCTIC dataset. Note that different from the `egs/arctic/bv2` recipe, we followed previous works [1, 2] and used a 1032/50/50 train/dev/test split.

If you are familiar with this toolkit or the ESPNet series, please note that `global_gain` is set to 0.95 for L2-ARCTIC to normalize the waveform samples such that all values are within `[-1, 1]`.

1. G. Zhao, S. Ding, and R. Gutierrez-Osuna, “Converting Foreign Accent Speech Without a Reference,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2367–2381, 2021
2. W. Quamer, A. Das, J. Levis, E. Chukharev-Hudilainen, and R. GutierrezOsuna, “Zero-shot foreign accent conversion without a native reference,” in Proc. Interspeech 2022, 2022, pp. 4920–4924.

## Non-parallel frame-based VC model training

All three methods make use of a non-parallel frame-based VC model for speaker identity conversion. The model was trained beforehand using the  [S3PRL-VC](https://github.com/unilight/s3prl-vc) toolkit. For now, we only plan to provide the pre-trained model checkpoint. Please contact us if you want to implement any of the three methods using your own dataset.

## Citation

If you use this recipe, please consider citing the following paper:

```
@inproceedings{huang2022evaluating,
  title={{Evaluating Methods for Ground-Truth-Free Foreign Accent Conversion}},
  author={Huang, Wen-Chin and and Toda, Tomoki},
  booktitle={Proc. APSIPC ASC},
  year={2023}
}
```