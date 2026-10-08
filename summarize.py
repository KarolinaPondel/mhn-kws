"""Mean +/- std over seeds of accuracy (%) per paradigm / model / noise condition.

    python summarize.py hast/runs [--metric accuracy] [--snr 20 10 0 -5 -10]

Reads runs/<paradigm>/<model>/seed_<k>/noise_snr_{metrics,accuracy}.csv (columns: noise_type, snr_db, <metric>...).
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs")
p.add_argument("--metric", default="accuracy", help="accuracy | macro_f1 | macro_precision | macro_recall | macro_roc_auc")
p.add_argument("--snr", type=int, nargs="+", default=[20, 10, 0, -5, -10])
args = p.parse_args()

values = defaultdict(lambda: defaultdict(list))  # (paradigm, model) -> condition -> [per-seed values]
for path in glob.glob(os.path.join(args.runs, "*", "*", "seed_*", "noise_snr_*.csv")):
    paradigm, model = path.split(os.sep)[-4:-2]
    with open(path) as f:
        for row in csv.DictReader(f):
            values[(paradigm, model)][(row["noise_type"], row["snr_db"])].append(float(row[args.metric]))

conditions = [("clean", "inf")] + [(n, str(s)) for n in ("white", "pink", "babble") for s in args.snr]
print(f"{args.metric} (%), mean +/- std over seeds\n")
for paradigm in sorted({k[0] for k in values}):
    print(f"## {paradigm}\n")
    print("| model | n | " + " | ".join("clean" if c[0] == "clean" else f"{c[0]} {c[1]} dB" for c in conditions) + " |")
    print("|---|---|" + "---|" * len(conditions))
    for (par, model), cells in sorted(values.items()):
        if par != paradigm:
            continue
        n = len(cells[("clean", "inf")])
        print(f"| {model} | {n} | " + " | ".join(
            f"{100 * np.mean(cells[c]):.2f} +/- {100 * np.std(cells[c], ddof=1):.2f}" if len(cells[c]) > 1 else "-"
            for c in conditions) + " |")
    print()
