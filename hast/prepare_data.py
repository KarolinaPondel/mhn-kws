"""Download Speech Commands v0.02 and write ``data/datafiles/{train,valid,test}.json``.

Uses the official validation / testing lists; train = everything else (35 keyword classes,
``_background_noise_`` excluded). NoiseX-92 is not downloaded automatically (see README).
"""

import csv
import json
import os
import tarfile
import urllib.request

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(DATA, "speech_commands_v0.02")

if not os.path.isdir(ROOT):
    os.makedirs(ROOT)
    archive = os.path.join(DATA, "speech_commands_v0.02.tar.gz")
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(ROOT)
    os.remove(archive)


def read_list(name):
    with open(os.path.join(ROOT, name)) as f:
        return [line.strip() for line in f if line.strip()]


valid, test = read_list("validation_list.txt"), read_list("testing_list.txt")
held_out = set(valid) | set(test)
train = sorted(
    f"{word}/{wav}" for word in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, word))
    and word != "_background_noise_" for wav in os.listdir(os.path.join(ROOT, word))
    if f"{word}/{wav}" not in held_out)

with open(os.path.join(DATA, "labels.csv")) as f:
    mid_of = {row["display_name"]: row["mid"] for row in csv.DictReader(f)}  # word -> "/m/spcmdNN"

os.makedirs(os.path.join(DATA, "datafiles"), exist_ok=True)
for split, files in (("train", train), ("valid", valid), ("test", test)):
    items = [{"wav": os.path.join(ROOT, f), "labels": mid_of[f.split("/")[0]]} for f in files]
    with open(os.path.join(DATA, "datafiles", f"{split}.json"), "w") as f:
        json.dump({"data": items}, f)
    print(split, len(items))
