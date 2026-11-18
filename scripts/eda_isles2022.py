"""Exploração inicial do dataset ISLES-2022 .

Gera:
- estatísticas descritivas das imagens e máscaras
- figuras para o capítulo de Resultados
- CSV com métricas por caso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ISLES-2022"
OUT_DIR = ROOT / "pratica" / "outputs"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"
THESIS_FIG = ROOT / "TCC1_extracted" / "figuras"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")


def list_subjects(data_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-strokecase")
    )


def load_nii(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms


def voxel_volume_ml(mask: np.ndarray, zooms: tuple[float, float, float]) -> float:
    voxel_mm3 = float(np.prod(zooms))
    return float(mask.sum() * voxel_mm3 / 1000.0)


def mid_slice_with_lesion(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(0, 1)) if mask.ndim == 3 else mask.sum(axis=0)
    if np.all(counts == 0):
        return mask.shape[-1] // 2
    return int(np.argmax(counts))


def collect_case_stats(subject: str) -> dict:
    dwi_path = DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_dwi.nii.gz"
    adc_path = DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_adc.nii.gz"
    flair_path = DATA_DIR / subject / "ses-0001" / "anat" / f"{subject}_ses-0001_FLAIR.nii.gz"
    msk_path = DATA_DIR / "derivatives" / subject / "ses-0001" / f"{subject}_ses-0001_msk.nii.gz"

    dwi, zooms = load_nii(dwi_path)
    adc, _ = load_nii(adc_path)
    flair, flair_zooms = load_nii(flair_path)
    mask, mask_zooms = load_nii(msk_path)
    mask_bin = (mask > 0).astype(np.uint8)

    return {
        "subject": subject,
        "dwi_shape": str(dwi.shape),
        "adc_shape": str(adc.shape),
        "flair_shape": str(flair.shape),
        "mask_shape": str(mask.shape),
        "dwi_spacing_mm": f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f}",
        "flair_spacing_mm": f"{flair_zooms[0]:.2f}x{flair_zooms[1]:.2f}x{flair_zooms[2]:.2f}",
        "lesion_voxels": int(mask_bin.sum()),
        "lesion_volume_ml": voxel_volume_ml(mask_bin, mask_zooms),
        "lesion_slices": int(np.sum(mask_bin.sum(axis=(0, 1)) > 0)),
        "dwi_mean": float(np.mean(dwi)),
        "dwi_std": float(np.std(dwi)),
        "adc_mean": float(np.mean(adc)),
        "adc_std": float(np.std(adc)),
        "has_lesion": bool(mask_bin.sum() > 0),
    }


def save_example_overlay(subject: str, out_path: Path) -> None:
    dwi_path = DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_dwi.nii.gz"
    adc_path = DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_adc.nii.gz"
    flair_path = DATA_DIR / subject / "ses-0001" / "anat" / f"{subject}_ses-0001_FLAIR.nii.gz"
    msk_path = DATA_DIR / "derivatives" / subject / "ses-0001" / f"{subject}_ses-0001_msk.nii.gz"

    dwi, _ = load_nii(dwi_path)
    adc, _ = load_nii(adc_path)
    flair, _ = load_nii(flair_path)
    mask = (load_nii(msk_path)[0] > 0).astype(np.uint8)

    z = mid_slice_with_lesion(mask)
    z_flair = flair.shape[-1] // 2

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        (dwi[..., z], "DWI", None),
        (adc[..., z], "ADC", None),
        (flair[..., z_flair], "FLAIR", None),
        (dwi[..., z], "DWI + máscara", mask[..., z]),
    ]
    for ax, (img, title, ov) in zip(axes, panels):
        ax.imshow(np.rot90(img), cmap="gray")
        if ov is not None and ov.any():
            overlay = np.ma.masked_where(ov == 0, ov)
            ax.imshow(np.rot90(overlay), cmap="autumn", alpha=0.45)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Exemplo ISLES-2022 — {subject}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(THESIS_FIG / out_path.name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_volume_hist(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["lesion_volume_ml"], bins=15, kde=True, ax=ax, color="#2c6eaa")
    ax.set_xlabel("Volume da lesão (mL)")
    ax.set_ylabel("Número de casos")
    ax.set_title("Distribuição do volume das lesões (amostra ISLES-2022)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(THESIS_FIG / out_path.name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_modality_shapes(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["dwi_shape"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="barh", ax=ax, color="#3d8b6e")
    ax.set_xlabel("Número de casos")
    ax.set_ylabel("Dimensões DWI (H×W×Z)")
    ax.set_title("Variabilidade dimensional das imagens DWI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(THESIS_FIG / out_path.name, dpi=200, bbox_inches="tight")
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
    if not subjects:
        raise SystemExit(f"Nenhum sujeito encontrado em {DATA_DIR}")

    print(f"Analisando {len(subjects)} casos em {DATA_DIR}")
    rows = [collect_case_stats(s) for s in subjects]
    df = pd.DataFrame(rows)
    csv_path = MET_DIR / "eda_isles2022_sample.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_cases": int(len(df)),
        "n_with_lesion": int(df["has_lesion"].sum()),
        "lesion_volume_ml_mean": float(df["lesion_volume_ml"].mean()),
        "lesion_volume_ml_median": float(df["lesion_volume_ml"].median()),
        "lesion_volume_ml_min": float(df["lesion_volume_ml"].min()),
        "lesion_volume_ml_max": float(df["lesion_volume_ml"].max()),
        "unique_dwi_shapes": int(df["dwi_shape"].nunique()),
        "unique_flair_shapes": int(df["flair_shape"].nunique()),
    }
    (MET_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df_pos = df[df["has_lesion"]].sort_values("lesion_volume_ml")
    example = df_pos.iloc[len(df_pos) // 2]["subject"] if len(df_pos) else subjects[0]

    save_example_overlay(example, FIG_DIR / "isles2022_exemplo_modalidades.png")
    plot_volume_hist(df, FIG_DIR / "isles2022_distribuicao_volume.png")
    plot_modality_shapes(df, FIG_DIR / "isles2022_variabilidade_dimensoes.png")

    print("Resumo:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Exemplo visual: {example}")
    print(f"Figuras em: {FIG_DIR}")
    print(f"Métricas em: {MET_DIR}")


if __name__ == "__main__":
    main()
