"""Pieces shared by train.py and eval_noise.py."""

import numpy as np
import torch
import torchaudio
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from model import N_LABELS

MFCC = dict(sample_rate=16000, n_mfcc=40,
            melkwargs=dict(n_fft=480, hop_length=160, n_mels=40, f_min=20.0, f_max=4000.0, power=2.0))


def mfcc_extractor(device):
    """waveform (B, 16000) -> MFCC (B, 101, 40).

    Note: torchaudio's MFCC clamps the dB-scaled spectrogram at (batch maximum - 80 dB), so a clip's
    features depend slightly on the other clips in its batch. Kept as in the thesis runs; evaluation
    therefore uses fixed batch sizes (dev 16, test / noise sweep 256 / 128) and a fixed clip order.
    """
    extractor = torchaudio.transforms.MFCC(**MFCC).to(device)
    return lambda wave: extractor(wave).transpose(1, 2).contiguous()


def macro_metrics(y_true, y_pred, y_prob):
    """Macro F1 / precision / recall / one-vs-rest ROC AUC over all 32 output classes.

    Class 1 never occurs (no "unknown" class), so it contributes 0 to the macro F1 / precision /
    recall averages and is skipped for ROC AUC.
    """
    labels = list(range(N_LABELS))
    kw = dict(labels=labels, average="macro", zero_division=0)
    f1, precision, recall = (f1_score(y_true, y_pred, **kw), precision_score(y_true, y_pred, **kw),
                             recall_score(y_true, y_pred, **kw))
    try:
        auc = roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro")
    except ValueError:
        scores = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in labels
                  if 0 < (y_true == i).sum() < len(y_true)]
        auc = float(np.mean(scores))
    return dict(macro_f1=f1, macro_precision=precision, macro_recall=recall, macro_roc_auc=auc)


@torch.no_grad()
def predict(model, loader, to_mfcc, device):
    """-> (labels, predicted classes, softmax probabilities) as numpy arrays."""
    model.eval()
    labels, probs = [], []
    for wave, y in loader:
        probs.append(torch.softmax(model(to_mfcc(wave.to(device))), dim=1).cpu().numpy())
        labels.append(y.numpy())
    labels, probs = np.concatenate(labels), np.concatenate(probs)
    return labels, probs.argmax(1), probs
