"""Evaluate a trained model on the test set: clean and with NoiseX-92 white / pink / babble noise
at SNR -10 ... 40 dB (step 5). Writes ``<exp_dir>/noise_snr_accuracy.csv``."""

import argparse
import csv
import json
import os

import torch
from torch.utils import data

from common import macro_metrics, mfcc_extractor, predict
from data import SpeechDataset, collate, list_split, load_noise, NOISES
from model import SpeechModel

SNRS = range(-10, 41, 5)
FIELDS = ["noise_type", "snr_db", "accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_roc_auc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--data_dir", default="data/speech_commands")
    p.add_argument("--noise_dir", default="../data/NoiseX-92")
    p.add_argument("--num_workers", type=int, default=6)
    args = p.parse_args()

    with open(os.path.join(args.exp_dir, "config.json")) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechModel(cfg["aggregation"], cfg["steps"], supcon_dim=cfg["supcon_dim"] if cfg["paradigm"] == "supcon" else 0)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "model.pt"), map_location="cpu"))
    model.to(device)
    to_mfcc = mfcc_extractor(device)

    paths, labels = list_split(args.data_dir, "test")
    noise = load_noise(args.noise_dir)
    rows = []
    for noise_type, snr in [("clean", None)] + [(n, s) for n in NOISES for s in SNRS]:
        dataset = SpeechDataset(paths, labels, "eval", noise, eval_noise=None if snr is None else (noise_type, snr))
        loader = data.DataLoader(dataset, batch_size=128, num_workers=args.num_workers, collate_fn=collate)
        y_true, y_pred, y_prob = predict(model, loader, to_mfcc, device)
        row = dict(noise_type=noise_type, snr_db="inf" if snr is None else snr,
                   accuracy=(y_true == y_pred).mean(), **macro_metrics(y_true, y_pred, y_prob))
        rows.append({k: f"{v:.6f}" if isinstance(v, float) or hasattr(v, "dtype") else v for k, v in row.items()})
        print(f"{noise_type:6s} snr={row['snr_db']!s:>3}  acc={float(row['accuracy']):.4f}", flush=True)

    with open(os.path.join(args.exp_dir, "noise_snr_accuracy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
