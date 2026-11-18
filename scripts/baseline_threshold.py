
"""Baseline clássico de segmentação no ISLES-2022.

Usa limiar adaptativo em DWI + refinamentos morfológicos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ISLES-2022"
OUT_DIR = ROOT / "pratica" / "outputs"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"
THESIS_FIG = ROOT / "TCC1_extracted" / "figuras"
THESIS_FIG.mkdir(parents=True, exist_ok=True)

FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


def list_subjects(data_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-strokecase")
    )


def load_nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def normalize01(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    mn, mx = vol.min(), vol.max()
    if mx <= mn:
        return np.zeros_like(vol)
    return (vol - mn) / (mx - mn)


def baseline_segment(dwi: np.ndarray, adc: np.ndarray) -> np.ndarray:
    """Heurística: hiperintensidade em DWI + hipointensidade relativa em ADC."""
    dwi_n = normalize01(dwi)
    adc_n = normalize01(adc)

    thr_dwi = float(np.percentile(dwi_n, 97))
    cand = dwi_n >= thr_dwi

    thr_adc = float(np.percentile(adc_n[cand], 60)) if cand.any() else 0.5
    pred = cand & (adc_n <= thr_adc)

    pred = ndi.binary_opening(pred, structure=np.ones((2, 2, 2)))
    pred = ndi.binary_closing(pred, structure=np.ones((3, 3, 3)))

    labeled, nlab = ndi.label(pred)
    if nlab == 0:
        return pred.astype(np.uint8)
    sizes = ndi.sum(pred, labeled, index=range(1, nlab + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(20, 0.15 * max(sizes))]
    out = np.isin(labeled, keep)
    return out.astype(np.uint8)


def mid_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(0, 1))
    if np.all(counts == 0):
        return mask.shape[-1] // 2
    return int(np.argmax(counts))


def save_qualitative(subject: str, dwi: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    z = mid_slice(gt if gt.any() else pred)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[0].set_title("DWI")
    axes[1].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[1].imshow(np.rot90(np.ma.masked_where(gt[..., z] == 0, gt[..., z])), cmap="autumn", alpha=0.5)
    axes[1].set_title("Máscara especialista (GT)")
    axes[2].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[2].imshow(np.rot90(np.ma.masked_where(pred[..., z] == 0, pred[..., z])), cmap="winter", alpha=0.5)
    axes[2].set_title("Baseline (DWI+ADC)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Comparação qualitativa — {subject}", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "isles2022_baseline_qualitativo.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(THESIS_FIG / out.name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-subjects", type=int, default=40,
                   help="Primeiros N casos (padrão 40 = amostra do TCC I); 0 = todos")
    return p.parse_args()


def _select(subjects: list[str], n: int) -> list[str]:
    return subjects[:n] if n > 0 else subjects


def main() -> None:
    args = parse_args()
    subjects = _select(list_subjects(DATA_DIR), args.max_subjects)
    rows = []
    best = None

    for subject in subjects:
        dwi = load_nii(DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_dwi.nii.gz")
        adc = load_nii(DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_adc.nii.gz")
        gt = (load_nii(DATA_DIR / "derivatives" / subject / "ses-0001" / f"{subject}_ses-0001_msk.nii.gz") > 0).astype(np.uint8)

       
        if dwi.shape != gt.shape or adc.shape != gt.shape:
            rows.append({"subject": subject, "dice": np.nan, "iou": np.nan, "skipped": True})
            continue

        pred = baseline_segment(dwi, adc)
        d = dice_score(pred, gt)
        i = iou_score(pred, gt)
        rows.append({"subject": subject, "dice": d, "iou": i, "skipped": False})
        if best is None or d > best[0]:
            best = (d, subject, dwi, gt, pred)

    df = pd.DataFrame(rows)
    df.to_csv(MET_DIR / "baseline_metrics.csv", index=False)

    valid = df[~df["skipped"]].dropna()
    summary = {
        "n_evaluated": int(len(valid)),
        "dice_mean": float(valid["dice"].mean()) if len(valid) else None,
        "dice_std": float(valid["dice"].std()) if len(valid) else None,
        "dice_median": float(valid["dice"].median()) if len(valid) else None,
        "iou_mean": float(valid["iou"].mean()) if len(valid) else None,
        "iou_std": float(valid["iou"].std()) if len(valid) else None,
        "iou_median": float(valid["iou"].median()) if len(valid) else None,
        "method": "limiar adaptativo DWI + restrição ADC + morfologia",
    }
    (MET_DIR / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if best is not None:
        _, subject, dwi, gt, pred = best
        save_qualitative(subject, dwi, gt, pred)

 
    if len(valid):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid["dice"], bins=12, color="#c45c26", edgecolor="white")
        ax.axvline(summary["dice_mean"], color="black", linestyle="--", label=f"média={summary['dice_mean']:.3f}")
        ax.set_xlabel("Dice coefficient")
        ax.set_ylabel("Número de casos")
        ax.set_title("Desempenho do baseline clássico (amostra ISLES-2022)")
        ax.legend()
        fig.tight_layout()
        out = FIG_DIR / "isles2022_baseline_dice.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(THESIS_FIG / out.name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Baseline summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
