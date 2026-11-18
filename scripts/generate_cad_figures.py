"""Gera figuras comparativas estilo CAD para o monógrafo (TCC II).

Compara baseline × U-Net × GT no mesmo caso e painel multi-caso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_threshold import baseline_segment  # noqa: E402
from isles_common import FIG_DIR, MET_DIR, dice_score, load_case, mid_slice, postprocess_mask  # noqa: E402
from isles_dataset import predict_volume  # noqa: E402
from unet_model import UNet2D  # noqa: E402

THESIS_FIG = ROOT / "TCC2_Overleaf" / "figuras"


def load_unet(ckpt: Path, device: torch.device) -> tuple[UNet2D, int]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    model = UNet2D(2, 1, base=int(args.get("base_channels", 32)))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, int(args.get("size", 128))


def panel_case(subject: str, dwi, gt, base, unet, out: Path) -> None:
    z = mid_slice(gt if gt.any() else unet)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
    titles = ["DWI", "GT", f"Baseline\nDice={dice_score(base, gt):.2f}", f"U-Net\nDice={dice_score(unet, gt):.2f}"]
    masks = [None, gt, base, unet]
    cmaps = [None, "autumn", "cool", "winter"]
    for ax, title, m, cm in zip(axes, titles, masks, cmaps):
        ax.imshow(np.rot90(dwi[..., z]), cmap="gray")
        if m is not None:
            ax.imshow(np.rot90(np.ma.masked_where(m[..., z] == 0, m[..., z])), cmap=cm, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Comparativo CAD — {subject}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(THESIS_FIG / out.name.replace(".png", ".jpg"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = ROOT / "pratica/outputs/checkpoints/unet2d_isles_aug_best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "pratica/outputs/checkpoints/unet2d_isles_best.pt"
    model, size = load_unet(ckpt, device)

    # casos: melhor e pior do CSV detalhado, se existir
    import pandas as pd

    csv = MET_DIR / "unet_aug_metrics_detailed.csv"
    if not csv.exists():
        csv = MET_DIR / "unet_metrics_detailed.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = df[~df.get("skipped", False)] if "skipped" in df.columns else df
        best_s = df.loc[df["dice"].idxmax(), "subject"]
        worst_s = df.loc[df["dice"].idxmin(), "subject"]
        subjects = [best_s, worst_s]
    else:
        subjects = ["sub-strokecase0003", "sub-strokecase0034"]

    for subject in subjects:
        dwi, adc, gt = load_case(subject)
        base = baseline_segment(dwi, adc)
        unet = predict_volume(model, dwi, adc, size=size, device=device)
        unet_pp = postprocess_mask(unet)
        tag = "bom" if subject == subjects[0] else "dificil"
        panel_case(subject, dwi, gt, base, unet_pp, FIG_DIR / f"cad_comparativo_{tag}.png")
        print("figura", tag, subject)

    # mock de tela CAD (painel informativo)
    subject = subjects[0]
    dwi, adc, gt = load_case(subject)
    base = baseline_segment(dwi, adc)
    unet = postprocess_mask(predict_volume(model, dwi, adc, size=size, device=device))
    z = mid_slice(gt)
    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.18, 0.82])
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.set_title("CAD AVC — TCC II (prova de conceito)", fontsize=14, loc="left", pad=8)
    ax_t.text(
        0.01,
        0.2,
        f"Caso: {subject}  |  Método: U-Net 2D  |  Dice={dice_score(unet, gt):.3f}  |  "
        f"Baseline Dice={dice_score(base, gt):.3f}  |  Apoio ao especialista",
        fontsize=10,
        transform=ax_t.transAxes,
    )
    for i, (title, m, cm) in enumerate(
        [("DWI", None, None), ("Predição U-Net", unet, "winter"), ("Máscara especialista", gt, "autumn")]
    ):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(np.rot90(dwi[..., z]), cmap="gray")
        if m is not None:
            ax.imshow(np.rot90(np.ma.masked_where(m[..., z] == 0, m[..., z])), cmap=cm, alpha=0.5)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "cad_interface_mock.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    # jpeg for thesis
    from PIL import Image

    Image.open(out).convert("RGB").save(THESIS_FIG / "cad_interface_mock.jpg", "JPEG", quality=88)
    for name in ["cad_comparativo_bom.png", "cad_comparativo_dificil.png"]:
        p = FIG_DIR / name
        if p.exists():
            Image.open(p).convert("RGB").save(THESIS_FIG / name.replace(".png", ".jpg"), "JPEG", quality=88)
    plt.close(fig)
    print("OK →", THESIS_FIG)


if __name__ == "__main__":
    main()
