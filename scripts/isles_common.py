"""Utilitários compartilhados ISLES-2022 (I/O, métricas, caminhos)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ISLES-2022"
OUT_DIR = ROOT / "pratica" / "outputs"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"
CKPT_DIR = OUT_DIR / "checkpoints"

for _d in (FIG_DIR, MET_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def list_subjects(data_dir: Path = DATA_DIR) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-strokecase")
    )


def paths_for_subject(subject: str, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    ses = "ses-0001"
    return {
        "dwi": data_dir / subject / ses / "dwi" / f"{subject}_{ses}_dwi.nii.gz",
        "adc": data_dir / subject / ses / "dwi" / f"{subject}_{ses}_adc.nii.gz",
        "mask": data_dir / "derivatives" / subject / ses / f"{subject}_{ses}_msk.nii.gz",
    }


def load_nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def load_case(subject: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = paths_for_subject(subject, data_dir)
    dwi = load_nii(p["dwi"])
    adc = load_nii(p["adc"])
    gt = (load_nii(p["mask"]) > 0).astype(np.uint8)
    return dwi, adc, gt


def normalize01(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    mn, mx = float(vol.min()), float(vol.max())
    if mx <= mn:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - mn) / (mx - mn)


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


def mid_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(0, 1))
    if np.all(counts == 0):
        return mask.shape[-1] // 2
    return int(np.argmax(counts))


def split_subjects(
    subjects: list[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(subjects)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1) if n > 2 else 0
    n_test = n - n_train - n_val
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    if n_test == 0 and val:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}
