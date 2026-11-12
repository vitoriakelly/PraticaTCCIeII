"""Treino da U-Net 2D no ISLES-2022.

Exemplos:
  python pratica/scripts/train_unet.py --epochs 2 --max-subjects 8   # smoke
  python pratica/scripts/train_unet.py --epochs 25                   # treino completo (40 casos)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isles_common import CKPT_DIR, DATA_DIR, FIG_DIR, MET_DIR, list_subjects, split_subjects
from isles_dataset import IslesSliceDataset
from unet_model import UNet2D, bce_dice_loss


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def batch_dice(logits: torch.Tensor, masks: torch.Tensor, thr: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= thr).float()
    masks = masks.float()
    inter = (preds * masks).sum().item()
    denom = preds.sum().item() + masks.sum().item()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total_dice += batch_dice(logits.detach(), masks) * bs
            n += bs
    return {"loss": total_loss / max(n, 1), "dice": total_dice / max(n, 1)}


def plot_history(history: dict[str, list[float]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="treino")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[1].plot(epochs, history["train_dice"], label="treino")
    axes[1].plot(epochs, history["val_dice"], label="val")
    axes[1].set_title("Dice (fatias)")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino U-Net ISLES-2022")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--max-subjects", type=int, default=0, help="0 = todos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-name", type=str, default="unet2d_isles")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    subjects = list_subjects(DATA_DIR)
    if not subjects:
        raise SystemExit(f"Nenhum sujeito em {DATA_DIR}")
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    splits = split_subjects(subjects, seed=args.seed)
    print(f"Device: {device}")
    print(
        f"Subjects: {len(subjects)} | "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    train_ds = IslesSliceDataset(splits["train"], size=args.size, seed=args.seed)
    val_ds = IslesSliceDataset(splits["val"] or splits["train"][:1], size=args.size, seed=args.seed + 1)
    print(f"Fatias: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet2D(in_channels=2, out_channels=1, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val = float("inf")
    best_path = CKPT_DIR / f"{args.out_name}_best.pt"
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device)
        va = run_epoch(model, val_loader, None, device)
        scheduler.step(va["loss"])
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_dice"].append(tr["dice"])
        history["val_dice"].append(va["dice"])
        print(
            f"Época {epoch:02d}/{args.epochs}  "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.3f}  "
            f"val loss={va['loss']:.4f} dice={va['dice']:.3f}"
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "splits": splits,
                    "epoch": epoch,
                    "val_loss": va["loss"],
                    "val_dice": va["dice"],
                },
                best_path,
            )
            print(f"  ✓ checkpoint salvo em {best_path.name}")

    elapsed = time.time() - t0
    last_path = CKPT_DIR / f"{args.out_name}_last.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args), "splits": splits}, last_path)

    plot_history(history, FIG_DIR / f"{args.out_name}_history.png")
    meta = {
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
        "n_subjects": len(subjects),
        "splits": {k: len(v) for k, v in splits.items()},
        "best_checkpoint": str(best_path),
        "best_val_loss": best_val,
        "history": history,
        "args": vars(args),
    }
    (MET_DIR / f"{args.out_name}_train.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTreino concluído em {elapsed/60:.1f} min")
    print(f"Melhor modelo: {best_path}")


if __name__ == "__main__":
    main()
