# Modern Hopfield networks for noise-robust keyword spotting - code for the experiments

Code to reproduce the experiments of the thesis on keyword spotting on Speech Commands with
(modern) Hopfield networks, trained and evaluated under additive noise (NoiseX-92).

| directory | what | data |
|---|---|---|
| [`hast/`](hast) | **HAST** - Audio Spectrogram Transformer trained from scratch whose self-attention blocks are replaced by Hopfield layers, vs. the **Transformer AST** twin | Speech Commands v0.02, 35 classes |
| [`honk/`](honk) | **HONK** - small CNN keyword spotter with five sequence-aggregation heads: `baseline`, `att_one_step`, `att_pure`, `mhn`, `mhnpl` | Speech Commands v0.01, 30 words + silence |

Both are trained under four paradigms and evaluated on the clean test set and with white / pink /
babble noise at SNR 40 ... -10 dB (accuracy, macro-F1, macro-precision, macro-recall, macro ROC AUC), 5 seeds each:

| paradigm | training |
|---|---|
| `clean` | cross-entropy on clean audio |
| `augment` | cross-entropy with NoiseX-92 noise augmentation |
| `nt_xent` | augmentation + NT-Xent contrastive loss |
| `supcon` | augmentation + supervised contrastive (SupCon) loss |

## Layout

```
data/NoiseX-92/            white.wav, pink.wav, babble.wav            (you provide, see below)
hast/  prepare_data.py     Speech Commands v0.02 -> data/datafiles/*.json
       run.sh              train + noise-evaluate one (model, paradigm) over 5 seeds
       src/{model,data,train,eval_noise}.py
honk/  download_data.py    Speech Commands v0.01 -> data/speech_commands/
       run.sh              train + noise-evaluate one (aggregation, paradigm) over 5 seeds
       src/{model,data,common,train,eval_noise}.py
summarize.py               mean +/- std over seeds -> markdown tables
```

All hyper-parameters live in one place per project: `PARADIGMS` / `SHARED` in `hast/src/train.py`
and `CFG` in `honk/src/train.py`.

## Setup

```bash
pip install -r hast/requirements.txt     # HAST: torch, torchaudio, timm==0.4.5, hopfield-layers
pip install -r honk/requirements.txt     # HONK: torch, torchaudio, hopfield-layers, soundfile, librosa, datasets
```

**NoiseX-92.** Download the NOISEX-92 recordings and place `white.wav`, `pink.wav`, `babble.wav`
(mono, any sample rate - they are resampled to 16 kHz) in `data/NoiseX-92/`. They are not redistributed here.

**Speech Commands.**
```bash
python hast/prepare_data.py     # v0.02, official validation/testing lists (84 843 / 9 981 / 11 005 clips)
python honk/download_data.py    # v0.01 from the HuggingFace hub
```

## Running

```bash
cd hast && ./run.sh hopfield supcon            # models: hopfield | transformer
cd honk && ./run.sh mhnpl nt_xent              # aggregations: baseline | att_one_step | att_pure | mhn | mhnpl
                                               # paradigms:    clean | augment | nt_xent | supcon
# GPU=1 SEEDS="0 1" ./run.sh ...   to pick the GPU / a subset of seeds;  STEPS=0 ./run.sh mhn clean  (HONK) for 0 Hopfield update steps
python summarize.py hast/runs                  # tables, accuracy by default
python summarize.py honk/runs --metric macro_roc_auc
```

The whole grid is 2 x 4 cells for HAST and 5 x 4 cells for HONK, 5 seeds each. On one RTX 8000 an epoch takes ~ 30 s (HONK) and 25 s - 2 min (HAST, data-loading bound).
Each run writes `runs/<paradigm>/<model>/seed_<k>/{config.json, model checkpoint, noise_snr_*.csv, logs}`.

## Notes

* **Models.** `hast/src/model.py` and `honk/src/model.py` are condensed re-implementations. Loaded with the
  trained checkpoints of the thesis runs they give numerically identical logits (max |delta| = 0; checked for HAST
  hopfield/transformer x clean/augment/supcon/nt_xent and for all five HONK heads), and the first training epochs
  follow the thesis loss curves. A fresh run initialises weights in a different RNG order, so single seeds will
  not match the thesis runs to the last digit - compare seed averages.
* **Noise evaluation.** The evaluation noise segment is chosen deterministically per clip (HONK: by hashing
  `<label>/<file>`, so it does not depend on where the data lives; HAST: fixed seed). Noise recordings are
  resampled to 16 kHz with the installed `librosa`/`torchaudio`, so waveforms differ marginally between versions.
* **HONK features.** MFCCs are computed on batches with `torchaudio.transforms.MFCC`, whose dB clamp
  (max - 80 dB) is taken over the whole batch, so a clip's features depend slightly on its batch companions.
  This is kept as in the thesis runs (batch sizes are fixed: train 256, dev 16, test 256, noise sweep 128), which
  means evaluation numbers move by ~ 0.1 pp if the clip order changes (e.g. 92.63 % vs 92.76 % clean accuracy for the
  same checkpoint with sorted vs. `os.listdir` order).
* **HONK** has no "unknown" class (those clips are dropped), so output 1 is never used; macro F1 /
  precision / recall are averaged over all 32 outputs and therefore include a zero for it, as in the thesis.
  `mhn`/`mhnpl` use 1 Hopfield update step by default (beta = 1.0 / 0.25).
* **HAST** is trained from scratch (no ImageNet / AudioSet weights): 4 blocks, 64-d, 2 heads, 16x16 patches with stride 10.
* The AST patch-embedding design follows [YuanGongND/ast](https://github.com/YuanGongND/ast) (BSD-3, see `LICENSE-AST-BSD3`);
  the keyword-spotting CNN follows [castorini/honk](https://github.com/castorini/honk) (MIT, see `LICENSE-honk-MIT`).
