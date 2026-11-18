"""Validação cruzada 5-fold por sujeito (U-Net 2D).

Exemplo (mais leve):
  python pratica/scripts/crossval_unet.py --epochs 12 --augment --folds 5
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from isles_common import CKPT_DIR, DATA_DIR, MET_DIR, dice_score, kfold_subject_splits, list_subjects, load_case
from isles_dataset import IslesSliceDataset, predict_volume
from train_unet import pick_device, run_epoch
from unet_model import UNet2D, bce_dice_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out-name", type=str, default="unet2d_cv")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = todos os casos disponíveis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    subjects = list_subjects(DATA_DIR)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    folds = kfold_subject_splits(subjects, n_folds=args.folds, seed=args.seed)
    print(f"Device={device} | subjects={len(subjects)} | folds={args.folds} | epochs={args.epochs}")

    fold_rows = []
    t_all = time.time()
    for fi, split in enumerate(folds):
        print(f"\n=== Fold {fi+1}/{args.folds} | train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} ===")
        train_ds = IslesSliceDataset(split["train"], size=args.size, seed=args.seed + fi, augment=args.augment)
        val_ds = IslesSliceDataset(split["val"] or split["train"][:1], size=args.size, seed=args.seed + 100 + fi)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = UNet2D(in_channels=2, out_channels=1, base=args.base_channels).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_state = None
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, opt, device)
            va = run_epoch(model, val_loader, None, device)
            print(f"  ep {epoch:02d} train_dice={tr['dice']:.3f} val_loss={va['loss']:.3f} val_dice={va['dice']:.3f}")
            if va["loss"] < best_val:
                best_val = va["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        dices = []
        for subject in split["test"]:
            dwi, adc, gt = load_case(subject)
            if dwi.shape != gt.shape:
                continue
            pred = predict_volume(model, dwi, adc, size=args.size, device=device)
            d = dice_score(pred, gt)
            dices.append(d)
            fold_rows.append({"fold": fi + 1, "subject": subject, "dice": d})
            print(f"  test {subject}: Dice={d:.3f}")

        ckpt_path = CKPT_DIR / f"{args.out_name}_fold{fi+1}.pt"
        torch.save(
            {"model_state": model.state_dict(), "args": vars(args), "splits": split, "fold": fi + 1},
            ckpt_path,
        )
        print(f"  fold Dice médio={np.mean(dices):.3f} | salvo {ckpt_path.name}")

    df = pd.DataFrame(fold_rows)
    df.to_csv(MET_DIR / f"{args.out_name}_folds.csv", index=False)
    summary = {
        "n_subjects": len(subjects),
        "folds": args.folds,
        "epochs": args.epochs,
        "augment": args.augment,
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std()),
        "dice_median": float(df["dice"].median()),
        "per_fold_mean": {
            str(int(f)): float(g["dice"].mean()) for f, g in df.groupby("fold")
        },
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (MET_DIR / f"{args.out_name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nCV resumo:", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
