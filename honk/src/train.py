"""Train the honk CNN keyword spotter with one sequence-aggregation head and one training paradigm.

    python src/train.py --aggregation mhnpl --paradigm supcon --seed 0 --exp_dir runs/supcon/mhnpl/seed_0

Paradigms:
    clean    cross-entropy on clean audio
    augment  cross-entropy, every training clip mixed with NoiseX-92 noise (SNR -10..40 dB)
    nt_xent  augment + NT-Xent between the clean-path and a noisier copy of each clip
    supcon   augment + supervised contrastive loss on a projection of the embedding
"""

import argparse
import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from common import mfcc_extractor, predict
from data import SpeechDataset, collate, list_split, load_noise
from model import AGGREGATIONS, SpeechModel

PARADIGMS = ("clean", "augment", "nt_xent", "supcon")
CFG = dict(
    n_epochs=60, batch_size=256, lr=0.003, momentum=0.9, weight_decay=1e-5,
    early_stopping_patience=7, early_stopping_min_delta=0.005,
    nt_xent_weight=0.5, nt_xent_temperature=0.07,
    supcon_weight=0.1, supcon_temperature=0.1, supcon_dim=64)


def nt_xent(z_clean, z_noisy, temperature):
    """Each clean embedding must pick its noisy twin among all clean (except itself) and noisy ones."""
    z_clean, z_noisy = F.normalize(z_clean, dim=1), F.normalize(z_noisy, dim=1)
    n = len(z_clean)
    sim = z_clean @ torch.cat([z_clean, z_noisy]).T / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    positive = sim[torch.arange(n), n + torch.arange(n)]
    self_mask = torch.cat([torch.eye(n, dtype=torch.bool, device=sim.device),
                           torch.zeros(n, n, dtype=torch.bool, device=sim.device)], dim=1)
    denominator = torch.exp(sim.masked_fill(self_mask, float("-inf"))).sum(dim=1)
    return -torch.log(torch.exp(positive) / (denominator + 1e-8) + 1e-16).mean()


def supcon(projections, labels, temperature):
    """Supervised contrastive loss: same-label samples in the batch are positives."""
    features = F.normalize(projections, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid = positives.any(dim=1)
    if not valid.any():
        return projections.sum() * 0.0
    log_prob = logits - torch.log(torch.exp(logits).masked_fill(self_mask, 0.0).sum(1, keepdim=True).clamp_min(1e-12))
    return -((positives.float() * log_prob).sum(1) / positives.sum(1).clamp_min(1))[valid].mean()


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id, seed):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregation", choices=AGGREGATIONS, required=True)
    p.add_argument("--paradigm", choices=PARADIGMS, required=True)
    p.add_argument("--steps", type=int, default=1, help="Hopfield update steps (mhn / mhnpl)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--data_dir", default="data/speech_commands")
    p.add_argument("--noise_dir", default="../data/NoiseX-92")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--n_epochs", type=int, default=CFG["n_epochs"], help="override for smoke tests")
    args = p.parse_args()
    cfg = {**CFG, **vars(args)}
    os.makedirs(args.exp_dir, exist_ok=True)
    with open(os.path.join(args.exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy = args.paradigm != "clean"
    noise = load_noise(args.noise_dir) if noisy else None
    dual = args.paradigm == "nt_xent"

    (tr_paths, tr_labels), (dev_paths, dev_labels), (te_paths, te_labels) = (
        list_split(args.data_dir, s) for s in ("train", "validation", "test"))
    train_set = SpeechDataset(tr_paths, tr_labels, "train", noise, train_noise=noisy, dual_path=dual)
    dev_set = SpeechDataset(dev_paths, dev_labels, "eval")
    test_set = SpeechDataset(te_paths, te_labels, "eval")

    # class-balanced sampling (inverse class frequency), including the synthetic silence clips
    labels = train_set.all_labels()
    class_weight = np.where(np.bincount(labels, minlength=32) > 0, 1.0 / np.maximum(np.bincount(labels, minlength=32), 1), 0.0)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = data.WeightedRandomSampler(torch.as_tensor(class_weight[labels], dtype=torch.double),
                                         num_samples=len(train_set), replacement=True, generator=generator)
    kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
              persistent_workers=args.num_workers > 0)
    train_loader = data.DataLoader(train_set, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True,
                                   generator=generator, worker_init_fn=partial(seed_worker, seed=args.seed), **kw)
    dev_loader = data.DataLoader(dev_set, batch_size=16, **kw)
    test_loader = data.DataLoader(test_set, batch_size=256, **kw)

    model = SpeechModel(args.aggregation, args.steps,
                        supcon_dim=cfg["supcon_dim"] if args.paradigm == "supcon" else 0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    to_mfcc = mfcc_extractor(device)

    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        start, losses = time.time(), []
        for batch, y in train_loader:
            y = y.to(device)
            if dual:
                x, x_noisy = (to_mfcc(w.to(device)) for w in batch)
                logits, embedding, noisy_embedding = model(x, x_noisy=x_noisy)
            else:
                x = to_mfcc(batch.to(device))
                logits, embedding, _ = model(x, return_embedding=True)
            loss = criterion(logits, y)
            if dual:
                loss = loss + cfg["nt_xent_weight"] * nt_xent(embedding, noisy_embedding, cfg["nt_xent_temperature"])
            elif args.paradigm == "supcon":
                loss = loss + cfg["supcon_weight"] * supcon(model.supcon_head(embedding), y, cfg["supcon_temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            dev_losses = [criterion(model(to_mfcc(w.to(device))), y.to(device)).item() for w, y in dev_loader]
        dev_loss = float(np.mean(dev_losses))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), dev_loss=dev_loss))
        print(f"epoch {epoch:2d}  train {history[-1]['train_loss']:.4f}  dev {dev_loss:.4f}  "
              f"({time.time() - start:.0f}s)", flush=True)

        if dev_loss < best_loss - cfg["early_stopping_min_delta"]:  # early stopping on dev cross-entropy
            best_loss, bad_epochs = dev_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.exp_dir, "model.pt"))
    y_true, y_pred, _ = predict(model, test_loader, to_mfcc, device)
    accuracy = float((y_true == y_pred).mean())
    print(f"test accuracy {accuracy:.4f}")
    with open(os.path.join(args.exp_dir, "result.json"), "w") as f:
        json.dump(dict(test_accuracy=accuracy, epochs=len(history), history=history), f, indent=1)


if __name__ == "__main__":
    main()
