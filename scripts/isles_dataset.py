"""Dataset 2D (fatias) do ISLES-2022 para treino da U-Net."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from isles_common import load_case, normalize01


def resize2d(img: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona H×W (ou H×W×C) para size×size."""
    if img.ndim == 2:
        factors = (size / img.shape[0], size / img.shape[1])
        return zoom(img, factors, order=1).astype(np.float32)
    raise ValueError(f"Esperado 2D, recebeu shape={img.shape}")


def prepare_volume(dwi: np.ndarray, adc: np.ndarray) -> np.ndarray:
    """Retorna volume (2, H, W, Z) normalizado."""
    dwi_n = normalize01(dwi)
    adc_n = normalize01(adc)
    return np.stack([dwi_n, adc_n], axis=0).astype(np.float32)


@dataclass
class SliceIndex:
    subject: str
    z: int
    has_lesion: bool


def augment_slice(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Augmentation leve 2D: flip e rotação 90° (x: C×H×W, y: H×W)."""
    if rng.random() < 0.5:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=1).copy()
    if rng.random() < 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=0).copy()
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
        y = np.rot90(y, k=k, axes=(0, 1)).copy()
    return x.astype(np.float32), y.astype(np.float32)


class IslesSliceDataset(Dataset):
    """Fatias axiais redimensionadas; prioriza fatias com lesão."""

    def __init__(
        self,
        subjects: list[str],
        size: int = 128,
        include_empty_ratio: float = 0.25,
        seed: int = 42,
        cache_volumes: bool = True,
        augment: bool = False,
    ) -> None:
        self.size = size
        self.cache_volumes = cache_volumes
        self.augment = augment
        self._rng = np.random.default_rng(seed + 17)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.indices: list[SliceIndex] = []

        rng = np.random.default_rng(seed)
        for subject in subjects:
            dwi, adc, gt = load_case(subject)
            if dwi.shape != adc.shape or dwi.shape != gt.shape:
                continue
            vol = prepare_volume(dwi, adc)
            if cache_volumes:
                self._cache[subject] = (vol, gt.astype(np.uint8))

            pos, neg = [], []
            for z in range(gt.shape[-1]):
                has = bool(gt[..., z].any())
                (pos if has else neg).append(SliceIndex(subject, z, has))

            keep_neg = []
            if pos and include_empty_ratio > 0 and neg:
                n_neg = max(1, int(round(len(pos) * include_empty_ratio)))
                n_neg = min(n_neg, len(neg))
                chosen = rng.choice(len(neg), size=n_neg, replace=False)
                keep_neg = [neg[i] for i in chosen]
            elif not pos and neg:
                # caso sem lesão anotada: amostra poucas fatias centrais
                mid = len(neg) // 2
                keep_neg = neg[max(0, mid - 2) : mid + 3]

            self.indices.extend(pos + keep_neg)

        rng.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_vol(self, subject: str) -> tuple[np.ndarray, np.ndarray]:
        if subject in self._cache:
            return self._cache[subject]
        dwi, adc, gt = load_case(subject)
        vol = prepare_volume(dwi, adc)
        return vol, gt.astype(np.uint8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        item = self.indices[idx]
        vol, gt = self._get_vol(item.subject)
        x = vol[:, :, :, item.z]  # (2, H, W)
        y = gt[:, :, item.z].astype(np.float32)

        x0 = resize2d(x[0], self.size)
        x1 = resize2d(x[1], self.size)
        y_r = resize2d(y, self.size)
        y_r = (y_r >= 0.5).astype(np.float32)
        x_np = np.stack([x0, x1], axis=0)
        if self.augment:
            x_np, y_r = augment_slice(x_np, y_r, self._rng)
            y_r = (y_r >= 0.5).astype(np.float32)

        x_t = torch.from_numpy(np.ascontiguousarray(x_np))
        y_t = torch.from_numpy(np.ascontiguousarray(y_r[None, ...]))
        return {
            "image": x_t,
            "mask": y_t,
            "subject": item.subject,
            "z": item.z,
        }


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    dwi: np.ndarray,
    adc: np.ndarray,
    size: int = 128,
    thr: float = 0.5,
    device: torch.device | None = None,
) -> np.ndarray:
    """Inferência fatia a fatia; devolve máscara no shape original."""
    device = device or next(model.parameters()).device
    model.eval()
    vol = prepare_volume(dwi, adc)
    _, h, w, zmax = vol.shape
    pred = np.zeros((h, w, zmax), dtype=np.uint8)

    for z in range(zmax):
        x0 = resize2d(vol[0, :, :, z], size)
        x1 = resize2d(vol[1, :, :, z], size)
        t = torch.from_numpy(np.stack([x0, x1], axis=0))[None, ...].to(device)
        logits = model(t)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        # volta ao tamanho original
        factors = (h / prob.shape[0], w / prob.shape[1])
        prob_full = zoom(prob, factors, order=1)
        # corrige possíveis 1-pixel de diferença por arredondamento
        prob_full = prob_full[:h, :w]
        if prob_full.shape != (h, w):
            tmp = np.zeros((h, w), dtype=np.float32)
            hh, ww = min(h, prob_full.shape[0]), min(w, prob_full.shape[1])
            tmp[:hh, :ww] = prob_full[:hh, :ww]
            prob_full = tmp
        pred[:, :, z] = (prob_full >= thr).astype(np.uint8)

    return pred
