"""Download Speech Commands v0.01 (HuggingFace) into ``data/speech_commands/{train,validation,test}/<label>/*.wav``."""

import os

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "speech_commands")

for split, examples in load_dataset("google/speech_commands", "v0.01").items():
    for ex in tqdm(examples, desc=split):
        folder = os.path.join(OUT, split, str(ex["label"]))
        os.makedirs(folder, exist_ok=True)
        sf.write(os.path.join(folder, os.path.basename(ex["file"])), ex["audio"]["array"], ex["audio"]["sampling_rate"])
