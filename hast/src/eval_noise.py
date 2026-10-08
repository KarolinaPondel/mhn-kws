"""Evaluate a trained model on the Speech Commands test set, clean and mixed with NoiseX-92
white / pink / babble noise at SNR 40 ... -10 dB. Writes ``<exp_dir>/noise_snr_metrics.csv``."""

import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from data import SpeechCommands
from train import build_model, set_seed

NOISES = ("pink", "white", "babble")
SNRS = (40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10)
FIELDS = ["noise_type", "snr_db", "accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_roc_auc"]


@torch.no_grad()
def metrics(model, loader, device):
    y_true, y_prob = [], []
    for x, y in loader:
        y_prob.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy())
        y_true.append(y.numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    y_pred = y_prob.argmax(1)
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        macro_roc_auc=roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--noise_dir", default="../data/NoiseX-92")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    with open(os.path.join(args.exp_dir, "config.json")) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt"), map_location=device))
    model.eval()

    d = args.data_dir

    def loader(noise=None, snr=0.0):
        ds = SpeechCommands(
            os.path.join(d, "datafiles/test.json"), os.path.join(d, "labels.csv"),
            cfg["dataset_mean"], cfg["dataset_std"], cfg["audio_length"],
            noise_root=args.noise_dir, noise_types=(noise,) if noise else (),
            noise_prob=1.0 if noise else 0.0, snr_min=snr, snr_max=snr)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    conditions = [("clean", "inf", None, 0.0)] + [(n, s, n, s) for n in NOISES for s in SNRS]
    for name, snr_label, noise, snr in conditions:
        set_seed(0)  # same noise segments for every model
        m = metrics(model, loader(noise, snr), device)
        rows.append(dict(noise_type=name, snr_db=snr_label, **{k: f"{v:.6f}" for k, v in m.items()}))
        print(f"{name:7s} snr={snr_label:>4}  acc={m['accuracy']:.4f}  auc={m['macro_roc_auc']:.4f}", flush=True)

    with open(os.path.join(args.exp_dir, "noise_snr_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
