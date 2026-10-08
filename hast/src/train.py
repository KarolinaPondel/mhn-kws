"""Train HAST (Hopfield AST) or a Transformer AST from scratch on Speech Commands v0.02.

    python src/train.py --model_type hopfield --paradigm supcon --seed 0 --exp_dir runs/supcon/hopfield/seed_0

Paradigms (the four training regimes compared in the thesis):
    clean    cross-entropy on clean audio
    augment  cross-entropy with NoiseX-92 noise augmentation
    supcon   augment + supervised contrastive loss (one view)
    nt_xent  augment + SimCLR NT-Xent loss (two independently noised views)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import NOISE_TYPES, SpeechCommands
from model import ASTModel

# ---------------------------------------------------------------------------------------------
# Hyper-parameters (tuned per paradigm for the Hopfield model; the Transformer twin reuses the
# optimiser settings and the shared block settings, with attention dropout 0).
# ---------------------------------------------------------------------------------------------
SHARED = dict(
    n_epochs=80, batch_size=64, depth=4, num_heads=2, dim=64, mlp_ratio=3.0, proj_drop=0.15,
    drop_path=0.08, hopfield_max_update_steps=2, fstride=10, tstride=10, audio_length=128,
    dataset_mean=-6.845978, dataset_std=5.5654526, early_stopping_patience=7,
    early_stopping_min_delta=0.005, grad_clip=0.5, label_smoothing=0.1,
    noise_prob=0.5, noise_snr_min=-10.0, noise_snr_max=40.0,
    contrastive_lambda=0.1, contrastive_temperature=0.1, contrastive_projection_dim=64)

PARADIGMS = {
    "clean":   dict(lr=5e-4,   weight_decay=1e-5, noise=False, contrastive=None,
                    hopfield_scaling=1.5, hopfield_init_values=0.1),
    "augment": dict(lr=2.5e-4, weight_decay=1e-5, noise=True,  contrastive=None,
                    hopfield_scaling=1.5, hopfield_init_values=0.1),
    "supcon":  dict(lr=5e-4,   weight_decay=5e-4, noise=True,  contrastive="supcon",
                    hopfield_scaling=1.0, hopfield_init_values=0.2),
    "nt_xent": dict(lr=5e-4,   weight_decay=5e-4, noise=True,  contrastive="nt_xent",
                    hopfield_scaling=1.0, hopfield_init_values=0.2),
}


def supervised_contrastive_loss(projections, labels, temperature):
    """SupCon: same-label samples in the batch are positives, all others negatives."""
    features = F.normalize(projections, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    n = labels.numel()
    self_mask = torch.eye(n, dtype=torch.bool, device=labels.device)
    positives = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid = positives.any(dim=1)
    if not valid.any():
        return projections.sum() * 0.0
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positives.float() * log_prob).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def nt_xent_loss(z1, z2, temperature):
    """SimCLR NT-Xent between two independently augmented views of the same utterances."""
    z1, z2 = F.normalize(z1.float(), dim=1), F.normalize(z2.float(), dim=1)
    n = z1.shape[0]
    sim = torch.cat([z1, z2]) @ torch.cat([z1, z2]).T / temperature
    sim = sim.masked_fill(torch.eye(2 * n, device=sim.device, dtype=torch.bool), -1e9)
    pos = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(sim.device)
    return F.cross_entropy(sim, pos)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg):
    return ASTModel(
        n_class=35, fstride=cfg["fstride"], tstride=cfg["tstride"], input_tdim=cfg["audio_length"],
        model_type=cfg["model_type"], dim=cfg["dim"], depth=cfg["depth"], num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"], proj_drop=cfg["proj_drop"],
        attn_drop=0.0, drop_path=cfg["drop_path"],
        hopfield_scaling=cfg["hopfield_scaling"],
        hopfield_max_update_steps=cfg["hopfield_max_update_steps"],
        hopfield_init_values=cfg["hopfield_init_values"],
        contrastive_lambda=cfg["contrastive_lambda"] if cfg["contrastive"] else 0.0,
        contrastive_projection_dim=cfg["contrastive_projection_dim"])


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return float(np.mean(losses)), correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["hopfield", "transformer"], required=True)
    p.add_argument("--paradigm", choices=list(PARADIGMS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--data_dir", default="data", help="holds datafiles/*.json and labels.csv")
    p.add_argument("--noise_dir", default="../data/NoiseX-92")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=None, help="override (e.g. 1 for a smoke test)")
    args = p.parse_args()

    cfg = {**SHARED, **PARADIGMS[args.paradigm], "model_type": args.model_type,
           "paradigm": args.paradigm, "seed": args.seed}
    if args.n_epochs:
        cfg["n_epochs"] = args.n_epochs
    os.makedirs(args.exp_dir, exist_ok=True)
    with open(os.path.join(args.exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = args.data_dir
    common = dict(label_csv=os.path.join(d, "labels.csv"), mean=cfg["dataset_mean"], std=cfg["dataset_std"],
                  target_length=cfg["audio_length"], noise_root=args.noise_dir,
                  noise_types=NOISE_TYPES if cfg["noise"] else (),
                  noise_prob=cfg["noise_prob"] if cfg["noise"] else 0.0,
                  snr_min=cfg["noise_snr_min"], snr_max=cfg["noise_snr_max"])
    two_views = cfg["contrastive"] == "nt_xent"
    train_set = SpeechCommands(os.path.join(d, "datafiles/train.json"), two_views=two_views, **common)
    val_set = SpeechCommands(os.path.join(d, "datafiles/valid.json"), **common)
    test_set = SpeechCommands(os.path.join(d, "datafiles/test.json"), **common)

    # class-balanced sampling (inverse class frequency)
    labels = np.array(train_set.labels())
    weights = (1.0 / (np.bincount(labels, minlength=35) + 1e-9))[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=True, prefetch_factor=3)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=2 * cfg["batch_size"], shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, batch_size=2 * cfg["batch_size"], shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=(0.95, 0.999), eps=1e-8)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-6)

    best_path = os.path.join(args.exp_dir, "best_model.pt")
    best_acc, best_val_loss, bad_epochs, step = -np.inf, np.inf, 0, 0
    history = []
    for epoch in range(1, cfg["n_epochs"] + 1):
        model.train()
        start, train_losses = time.time(), []
        for batch in train_loader:
            if two_views:
                x, x2, y = batch
                x2 = x2.to(device, non_blocking=True)
            else:
                x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if step <= 1000 and step % 50 == 0:  # linear warm-up over the first 1000 steps
                for group in optimizer.param_groups:
                    group["lr"] = step / 1000 * cfg["lr"]

            if cfg["contrastive"]:
                logits, proj = model(x, return_contrastive=True)
                loss = loss_fn(logits, y)
                if two_views:
                    _, proj2 = model(x2, return_contrastive=True)
                    con = nt_xent_loss(proj, proj2, cfg["contrastive_temperature"])
                else:
                    con = supervised_contrastive_loss(proj, y, cfg["contrastive_temperature"])
                loss = loss + cfg["contrastive_lambda"] * con
            else:
                loss = loss_fn(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
            step += 1

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=float(np.mean(train_losses)), val_loss=val_loss,
                            val_acc=val_acc, lr=optimizer.param_groups[0]["lr"]))
        print(f"epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}  "
              f"acc {val_acc:.4f}  ({time.time() - start:.0f}s)", flush=True)

        if val_acc > best_acc:  # checkpoint selection: best validation accuracy
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
        if best_val_loss - val_loss >= cfg["early_stopping_min_delta"]:  # early stopping: validation loss
            best_val_loss, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    with open(os.path.join(args.exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, val_acc = evaluate(model, val_loader, loss_fn, device)
    _, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"best checkpoint: val acc {val_acc:.4f}, test acc {test_acc:.4f}")
    with open(os.path.join(args.exp_dir, "eval_result.json"), "w") as f:
        json.dump(dict(val_acc=val_acc, test_acc=test_acc), f)


if __name__ == "__main__":
    main()
