"""Speech Commands v0.01 keyword spotting data: 30 words + silence (32 outputs, label 1 unused).

``data/{train,validation,test}/<label>/*.wav`` is produced by ``download_data.py`` (HF label ids
0..29 = the 30 words; folder 30 holds the unused ``_silence_`` / background class and is skipped).
Class 0 is synthetic silence (10 % of the split size), class ``i + 2`` is ``WORDS[i]``.
"""

import hashlib
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils import data

WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two",
         "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy",
         "house", "marvin", "sheila", "tree", "wow"]
N_LABELS = len(WORDS) + 2
NOISES = ("white", "pink", "babble")
SAMPLE_RATE = 16000
SILENCE_PROB = 0.1
TIMESHIFT_MS = 100
CACHE_SIZE = 32768
SNR_TRAIN_RANGE = (-10.0, 40.0)


def list_split(data_dir, split):
    """-> (wav paths, labels) for one split."""
    paths, labels = [], []
    for folder in sorted(os.listdir(os.path.join(data_dir, split))):
        if not folder.isdigit() or int(folder) >= len(WORDS):
            continue
        for name in sorted(os.listdir(os.path.join(data_dir, split, folder))):
            paths.append(os.path.join(data_dir, split, folder, name))
            labels.append(int(folder) + 2)
    return paths, labels


def load_noise(noise_dir):
    """NoiseX-92 recordings resampled to 16 kHz mono."""
    import librosa
    return {name: librosa.load(os.path.join(noise_dir, f"{name}.wav"), sr=SAMPLE_RATE)[0]
            for name in NOISES}


def mix_at_snr(clean, noise, snr_db):
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    scaled = noise * (clean_rms / (10 ** (snr_db / 20.0))) / (noise_rms + 1e-12)
    return np.clip(clean + scaled, -1.0, 1.0)


class _Cache(dict):
    """Insert-only cache with a size limit (as in the original honk loader)."""

    def __setitem__(self, key, value):
        if key in self or len(self) < CACHE_SIZE:
            super().__setitem__(key, value)


class SpeechDataset(data.Dataset):
    """Waveform dataset. ``mode``: 'train' | 'eval'.

    train: random time shift, NoiseX-92 noise at a random SNR in [-10, 40] dB when ``train_noise``
           (every sample, random type), and with ``dual_path`` also returns a second, additionally
           noised copy (for the NT-Xent clean/noisy pair).
    eval:  no augmentation; with ``eval_noise=(type, snr_db)`` a deterministic NoiseX-92 segment
           (picked by hashing the file name) is mixed in at the given SNR.
    """

    def __init__(self, paths, labels, mode, noise_waves=None, train_noise=False, dual_path=False,
                 eval_noise=None):
        self.paths, self.labels, self.mode = paths, labels, mode
        self.noise_waves = noise_waves or {}
        self.train_noise, self.dual_path, self.eval_noise = train_noise, dual_path, eval_noise
        self.n_silence = int(SILENCE_PROB * len(paths))
        self._audio, self._files = _Cache(), _Cache()

    def __len__(self):
        return len(self.paths) + self.n_silence

    def all_labels(self):
        return np.array(self.labels + [0] * self.n_silence, dtype=np.int64)

    def _noise_segment(self, name, length, key=None):
        wav = self.noise_waves[name]
        if len(wav) >= length:
            span = len(wav) - length
            start = (int(hashlib.sha1(f"{name}|{key}|{length}".encode()).hexdigest(), 16) % (span + 1)
                     if key is not None else random.randint(0, span))
            return wav[start:start + length]
        return np.tile(wav, int(np.ceil(length / len(wav))))[:length]

    def _random_noise(self, wave):
        name = random.choice(list(self.noise_waves))
        snr = random.uniform(*SNR_TRAIN_RANGE)
        return mix_at_snr(wave, self._noise_segment(name, len(wave)), snr)

    def _load(self, path):
        """path=None -> silence."""
        key = "/".join(path.split(os.sep)[-3:]) if path else "__silence__"  # path-independent id
        if random.random() < 0.7 or self.mode != "train":
            if key in self._audio:
                return self._audio[key]
        if path is None:
            wave = np.zeros(SAMPLE_RATE, dtype=np.float32)
        else:
            wave = self._files.get(key)
            if wave is None:
                wave, _ = sf.read(path, dtype="float32")
                self._files[key] = wave
        wave = np.pad(wave, (0, max(0, SAMPLE_RATE - len(wave))), "constant")
        if self.mode == "train":
            shift = random.randint(-SAMPLE_RATE * TIMESHIFT_MS // 1000, SAMPLE_RATE * TIMESHIFT_MS // 1000)
            a, b = -min(0, shift), max(0, shift)
            wave = np.pad(wave, (a, b), "constant")
            wave = wave[:len(wave) - a] if a else wave[b:]
            if self.train_noise:
                wave = self._random_noise(wave)
        elif self.eval_noise is not None:
            name, snr = self.eval_noise
            wave = mix_at_snr(wave, self._noise_segment(name, len(wave), key=key), snr)
        self._audio[key] = wave
        return wave

    def __getitem__(self, i):
        path, label = (self.paths[i], self.labels[i]) if i < len(self.paths) else (None, 0)
        clean = self._load(path)
        if self.dual_path and self.mode == "train":
            noisy = self._random_noise(clean.copy()) if self.train_noise else clean.copy()
            return clean, noisy, label
        return clean, label


def collate(batch):
    waves = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32))
    labels = torch.tensor([b[-1] for b in batch])
    if len(batch[0]) == 3:
        return (waves, torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.float32))), labels
    return waves, labels
