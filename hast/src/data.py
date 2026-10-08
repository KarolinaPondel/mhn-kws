"""Speech Commands v0.02 (35 classes) as log-mel filterbanks, with optional NoiseX-92 augmentation."""

import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

NOISE_TYPES = ("babble", "white", "pink")


def read_label_index(label_csv):
    with open(label_csv, "r") as f:
        return {row["mid"]: int(row["index"]) for row in csv.DictReader(f)}


def load_noise(noise_root, noise_types):
    """Load NoiseX-92 recordings (``<type>.wav``) as mono waveforms."""
    waveforms = {}
    for name in noise_types:
        path = os.path.join(noise_root, f"{name}.wav")
        if not os.path.exists(path):
            raise FileNotFoundError(f"NoiseX-92 file not found: {path} (see README, 'Data')")
        wav, sr = torchaudio.load(path)
        waveforms[name] = (wav.mean(dim=0, keepdim=True), sr)
    return waveforms


class SpeechCommands(Dataset):
    """One item = (fbank, label); with ``two_views`` = (fbank_a, fbank_b, label) for NT-Xent.

    With ``noise_prob`` > 0 every view is mixed (probability ``noise_prob``) with a random
    segment of a random NoiseX-92 recording at an SNR drawn uniformly from [snr_min, snr_max].
    """

    def __init__(self, json_path, label_csv, mean, std, target_length=128, num_mel_bins=128,
                 noise_root=None, noise_types=(), noise_prob=0.0, snr_min=0.0, snr_max=0.0,
                 two_views=False):
        with open(json_path, "r") as f:
            self.data = json.load(f)["data"]
        self.index = read_label_index(label_csv)
        self.mean, self.std = mean, std
        self.target_length, self.num_mel_bins = target_length, num_mel_bins
        self.noise_prob, self.snr_min, self.snr_max = noise_prob, snr_min, snr_max
        self.two_views = two_views
        self.noise = load_noise(noise_root, noise_types) if noise_prob > 0 and noise_types else {}

    def labels(self):
        return [self.index[d["labels"]] for d in self.data]

    def _add_noise(self, waveform, sr):
        name = random.choice(list(self.noise))
        noise, noise_sr = self.noise[name]
        if noise_sr != sr:
            noise = AF.resample(noise, noise_sr, sr)
            self.noise[name] = (noise, sr)
        if noise.shape[1] < waveform.shape[1]:
            noise = noise.repeat(1, int(np.ceil(waveform.shape[1] / noise.shape[1])))
        start = random.randint(0, max(noise.shape[1] - waveform.shape[1], 0))
        segment = noise[:, start:start + waveform.shape[1]]
        snr = torch.tensor([random.uniform(self.snr_min, self.snr_max)], dtype=waveform.dtype)
        return AF.add_noise(waveform, segment, snr)[0]

    def _fbank(self, wav_path):
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform - waveform.mean()
        if self.noise and random.random() < self.noise_prob:
            waveform = self._add_noise(waveform, sr)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # centre crop / zero-pad to target_length frames (10 ms hop)
        n = self.target_length * 160
        if waveform.shape[1] > n:
            start = (waveform.shape[1] - n) // 2
            waveform = waveform[:, start:start + n]
        elif waveform.shape[1] < n:
            left = (n - waveform.shape[1]) // 2
            waveform = F.pad(waveform, (left, n - waveform.shape[1] - left))

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type="hanning", num_mel_bins=self.num_mel_bins, dither=0.0, frame_shift=10)
        pad = self.target_length - fbank.shape[0]
        if pad > 0:
            fbank = F.pad(fbank, (0, 0, 0, pad))
        else:
            fbank = fbank[:self.target_length]
        return (fbank - self.mean) / (self.std * 2)

    def __getitem__(self, i):
        datum = self.data[i]
        label = torch.tensor(self.index[datum["labels"]], dtype=torch.long)
        view = self._fbank(datum["wav"])
        if not self.two_views:
            return view, label
        return view, self._fbank(datum["wav"]), label

    def __len__(self):
        return len(self.data)
