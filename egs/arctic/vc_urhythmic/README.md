# One-to-one seq2seq voice conversion with Urhythmic 

2024.03.

I would like to reiterate that **I did not propose or implement the Urhyrhmic method**. All credits belong to the original author's codebase: https://github.com/bshall/urhythmic.

Note: It is not really obvious from the original paper but Urhythmic is in fact a method for **one-to-one** voice conversion, i.e., training data is required for both source and target speakers. However the training dataset does not have to be parallel.

## Usage of this recipe

This is a really simple method. Training is only required for:

1. Rhythm model training(stage 4)
2. Vocoder fine-tuning (stage 5)

### Soft unit extraction, segmentation, rhythm model training

```
./run.sh --stage 0 --stop_stage 4
```

20 minutes should be enough for up to stage 4.

### Vocoder training

```
./run.sh --stage 5 --stop_stage 5
```

On a Tesla V100 it takes around 16-24 hours.

### Conversion and evaluation

```
./run.sh --stage 6 --stop_stage 7
```

## Comparison of Urhythmic and AAS-VC

I compared Urhythmic and AAS-VC using clb-slt test set with 932 and 80 training utterances.

| System    | Training utterances | MCD  | DDUR  | CER  | WER  |
|-----------|---------------------|------|-------|------|------|
| AAS-VC    | 932                 | 6.27 | 0.162 | 1.5  | 6.1  |
| AAS-VC    | 80                  | 7.02 | 0.232 | 12.0 | 25.9 |
| Urhythmic | 932                 | 6.92 | 0.204 | 0.5  | 3.4  |
| Urhythmic | 80                  | 6.86 | 0.247 | 0.6  | 3.4  |