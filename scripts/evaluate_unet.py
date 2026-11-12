"""Avalia checkpoint U-Net no split de teste (métricas volumétricas).

Exemplo:
  python pratica/scripts/evaluate_unet.py
  python pratica/scripts/evaluate_unet.py --checkpoint pratica/outputs/checkpoints/unet2d_isles_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from isles_common import (
    CKPT_DIR,
    DATA_DIR,
    FIG_DIR,
    MET_DIR,
    dice_score,
    iou_score,
    list_subjects,
    load_case,
    mid_slice,
    split_subjects,
)
from isles_dataset import predict_volume
from unet_model import UNet2D


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> tuple[UNet2D, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = UNet2D(
        in_channels=2,
        out_channels=1,
        base=int(args.get("base_channels", 32)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def save_qualitative(
    subject: str,
    dwi: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out: Path,
) -> None:
    z = mid_slice(gt if gt.any() else pred)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[0].set_title("DWI")
    axes[1].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[1].imshow(np.rot90(np.ma.masked_where(gt[..., z] == 0, gt[..., z])), cmap="autumn", alpha=0.5)
    axes[1].set_title("GT")
    axes[2].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[2].imshow(np.rot90(np.ma.masked_where(pred[..., z] == 0, pred[..., z])), cmap="winter", alpha=0.5)
    axes[2].set_title("U-Net")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"U-Net — {subject}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT_DIR / "unet2d_isles_best.pt",
    )
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--size", type=int, default=0, help="0 = usar size do checkpoint")
    p.add_argument("--thr", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint não encontrado: {args.checkpoint}")

    device = pick_device()
    model, ckpt = load_model(args.checkpoint, device)
    train_args = ckpt.get("args", {})
    size = args.size or int(train_args.get("size", 128))
    splits = ckpt.get("splits")
    if splits is None:
        subjects = list_subjects(DATA_DIR)
        if train_args.get("max_subjects"):
            subjects = subjects[: int(train_args["max_subjects"])]
        splits = split_subjects(subjects, seed=int(train_args.get("seed", 42)))

    if args.split == "all":
        eval_subjects = sorted(set(splits["train"] + splits["val"] + splits["test"]))
    else:
        eval_subjects = splits.get(args.split) or splits["test"]

    print(f"Device: {device} | split={args.split} | n={len(eval_subjects)} | size={size}")

    rows = []
    best = None
    for subject in eval_subjects:
        try:
            dwi, adc, gt = load_case(subject)
        except FileNotFoundError:
            rows.append({"subject": subject, "dice": np.nan, "iou": np.nan, "skipped": True})
            continue
        if dwi.shape != gt.shape or adc.shape != gt.shape:
            rows.append({"subject": subject, "dice": np.nan, "iou": np.nan, "skipped": True})
            continue
        pred = predict_volume(model, dwi, adc, size=size, thr=args.thr, device=device)
        d = dice_score(pred, gt)
        i = iou_score(pred, gt)
        rows.append({"subject": subject, "dice": d, "iou": i, "skipped": False})
        print(f"  {subject}: Dice={d:.3f} IoU={i:.3f}")
        if best is None or d > best[0]:
            best = (d, subject, dwi, gt, pred)

    df = pd.DataFrame(rows)
    out_csv = MET_DIR / "unet_metrics.csv"
    df.to_csv(out_csv, index=False)

    valid = df[~df["skipped"]].dropna()
    summary = {
        "n_evaluated": int(len(valid)),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "dice_mean": float(valid["dice"].mean()) if len(valid) else None,
        "dice_std": float(valid["dice"].std()) if len(valid) else None,
        "dice_median": float(valid["dice"].median()) if len(valid) else None,
        "iou_mean": float(valid["iou"].mean()) if len(valid) else None,
        "iou_std": float(valid["iou"].std()) if len(valid) else None,
        "iou_median": float(valid["iou"].median()) if len(valid) else None,
        "method": "U-Net 2D (DWI+ADC)",
    }
    (MET_DIR / "unet_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if best is not None:
        _, subject, dwi, gt, pred = best
        save_qualitative(subject, dwi, gt, pred, FIG_DIR / "unet_qualitativo.png")

    if len(valid):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid["dice"], bins=min(12, max(4, len(valid))), color="#2a6f97", edgecolor="white")
        ax.axvline(summary["dice_mean"], color="black", linestyle="--", label=f"média={summary['dice_mean']:.3f}")
        ax.set_xlabel("Dice coefficient")
        ax.set_ylabel("Número de casos")
        ax.set_title(f"Desempenho U-Net ({args.split})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "unet_dice_hist.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("\nResumo U-Net:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
