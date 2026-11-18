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


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    c = confusion_counts(pred, gt)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 1.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 1.0
    return float(prec), float(rec)


def lesion_volume_ml(mask: np.ndarray, zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    voxel_ml = (zooms[0] * zooms[1] * zooms[2]) / 1000.0
    return float(mask.astype(bool).sum() * voxel_ml)


def postprocess_mask(pred: np.ndarray, min_voxels: int = 20) -> np.ndarray:
    """Abertura/fechamento 3D + remoção de componentes muito pequenos."""
    from scipy import ndimage as ndi

    pred = pred.astype(bool)
    pred = ndi.binary_opening(pred, structure=np.ones((2, 2, 2)))
    pred = ndi.binary_closing(pred, structure=np.ones((3, 3, 3)))
    labeled, nlab = ndi.label(pred)
    if nlab == 0:
        return pred.astype(np.uint8)
    sizes = ndi.sum(pred, labeled, index=range(1, nlab + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    return np.isin(labeled, keep).astype(np.uint8)


def kfold_subject_splits(subjects: list[str], n_folds: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    """Gera folds por sujeito: cada fold tem train/val/test (val = 1/5 do train)."""
    rng = np.random.default_rng(seed)
    ids = list(subjects)
    rng.shuffle(ids)
    folds: list[dict[str, list[str]]] = []
    n = len(ids)
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1
    start = 0
    chunks = []
    for fs in fold_sizes:
        chunks.append(ids[start : start + fs])
        start += fs
    for i in range(n_folds):
        test = chunks[i]
        rest = [s for j, ch in enumerate(chunks) if j != i for s in ch]
        # 15% de rest como val
        n_val = max(1, int(round(len(rest) * 0.15))) if rest else 0
        val = rest[:n_val]
        train = rest[n_val:]
        folds.append({"train": train, "val": val, "test": test})
    return folds


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
