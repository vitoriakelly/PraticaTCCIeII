"""Interface CAD (prova de conceito) — baseline e U-Net no ISLES-2022.

Uso:
  # a partir da raiz do projeto
  source .venv/bin/activate
  streamlit run pratica/app/app_streamlit.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pratica" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from baseline_threshold import baseline_segment  # noqa: E402
from isles_common import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    dice_score,
    iou_score,
    list_subjects,
    load_nii,
    mid_slice,
    postprocess_mask,
)

st.set_page_config(
    page_title="CAD AVC — TCC",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

# Remove o botão Deploy e itens da barra superior (não fazem parte do CAD)
st.markdown(
    """
    <style>
    .stDeployButton, [data-testid="stToolbar"] {display: none !important; visibility: hidden !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_case(subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dwi = load_nii(DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_dwi.nii.gz")
    adc = load_nii(DATA_DIR / subject / "ses-0001" / "dwi" / f"{subject}_ses-0001_adc.nii.gz")
    gt_path = DATA_DIR / "derivatives" / subject / "ses-0001" / f"{subject}_ses-0001_msk.nii.gz"
    gt = (load_nii(gt_path) > 0).astype(np.uint8) if gt_path.exists() else None
    return dwi, adc, gt


def load_uploaded(file) -> np.ndarray:
    suffix = ".nii.gz" if file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        return np.asanyarray(nib.load(str(tmp_path)).dataobj)
    finally:
        tmp_path.unlink(missing_ok=True)


def volume_ml(mask: np.ndarray, zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    voxel_ml = (zooms[0] * zooms[1] * zooms[2]) / 1000.0
    return float(mask.astype(bool).sum() * voxel_ml)


@st.cache_resource
def load_unet(ckpt_path: str):
    import torch
    from isles_dataset import predict_volume
    from unet_model import UNet2D

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = UNet2D(in_channels=2, out_channels=1, base=int(args.get("base_channels", 32)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    size = int(args.get("size", 128))
    return model, device, size, predict_volume


def segment(
    method: str,
    dwi: np.ndarray,
    adc: np.ndarray,
    ckpt: Path | None,
    do_postprocess: bool = True,
) -> np.ndarray:
    if method == "Baseline (limiar)":
        return baseline_segment(dwi, adc)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError("Checkpoint U-Net não encontrado. Treine o modelo antes.")
    model, device, size, predict_volume = load_unet(str(ckpt))
    pred = predict_volume(model, dwi, adc, size=size, thr=0.5, device=device)
    if do_postprocess:
        pred = postprocess_mask(pred)
    return pred


def show_slice(dwi: np.ndarray, pred: np.ndarray, gt: np.ndarray | None, z: int, method: str) -> None:
    fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(11, 3.4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[0].set_title("DWI")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(dwi[..., z]), cmap="gray")
    axes[1].imshow(
        np.rot90(np.ma.masked_where(pred[..., z] == 0, pred[..., z])),
        cmap="winter",
        alpha=0.5,
    )
    axes[1].set_title(f"Predição ({method})")
    axes[1].axis("off")

    if gt is not None:
        axes[2].imshow(np.rot90(dwi[..., z]), cmap="gray")
        axes[2].imshow(
            np.rot90(np.ma.masked_where(gt[..., z] == 0, gt[..., z])),
            cmap="autumn",
            alpha=0.5,
        )
        axes[2].set_title("Máscara especialista (GT)")
        axes[2].axis("off")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    st.title("CAD — Apoio à segmentação de AVC isquêmico")
    st.caption(
        "Prova de conceito do TCC: baseline clássico e U-Net (DWI + ADC). "
        "Ferramenta de apoio — a decisão final é do especialista."
    )

    default_ckpt = CKPT_DIR / "unet2d_isles250_best.pt"
    if not default_ckpt.exists():
        default_ckpt = CKPT_DIR / "unet2d_isles_aug_best.pt"
    if not default_ckpt.exists():
        default_ckpt = CKPT_DIR / "unet2d_isles_best.pt"

    with st.sidebar:
        st.header("Configuração")
        method = st.radio("Método", ["Baseline (limiar)", "U-Net"], index=1)
        ckpt_file = st.text_input("Checkpoint U-Net", value=str(default_ckpt))
        ckpt = Path(ckpt_file)
        do_post = st.checkbox("Pós-processamento (U-Net)", value=True)
        if method == "U-Net" and not ckpt.exists():
            st.warning("Checkpoint ausente. Rode `train_unet.py` antes.")

        st.header("Entrada")
        mode = st.radio("Fonte dos dados", ["Caso ISLES-2022", "Upload NIfTI"], index=0)

        if mode == "Caso ISLES-2022":
            if not DATA_DIR.exists():
                st.error(f"Dataset não encontrado em `{DATA_DIR}`.")
                st.stop()
            subjects = list_subjects(DATA_DIR)
            if not subjects:
                st.error("Nenhum sujeito encontrado no ISLES-2022.")
                st.stop()
            subject = st.selectbox("Caso", subjects)
            if st.button("Carregar e segmentar", type="primary"):
                with st.spinner("Processando..."):
                    dwi, adc, gt = load_case(subject)
                    try:
                        pred = segment(method, dwi, adc, ckpt, do_postprocess=do_post)
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
                        st.stop()
                    st.session_state["case"] = {
                        "dwi": dwi,
                        "adc": adc,
                        "gt": gt,
                        "label": subject,
                        "pred": pred,
                        "method": method,
                    }
        else:
            up_dwi = st.file_uploader("DWI (.nii / .nii.gz)", type=["nii", "gz"])
            up_adc = st.file_uploader("ADC (.nii / .nii.gz)", type=["nii", "gz"])
            up_gt = st.file_uploader("Máscara GT (opcional)", type=["nii", "gz"])
            if st.button("Segmentar upload", type="primary"):
                if up_dwi is None or up_adc is None:
                    st.warning("Envie pelo menos DWI e ADC.")
                else:
                    with st.spinner("Processando..."):
                        dwi = load_uploaded(up_dwi)
                        adc = load_uploaded(up_adc)
                        gt = (load_uploaded(up_gt) > 0).astype(np.uint8) if up_gt else None
                        if dwi.shape != adc.shape:
                            st.error("DWI e ADC com dimensões diferentes.")
                        else:
                            try:
                                pred = segment(method, dwi, adc, ckpt, do_postprocess=do_post)
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))
                                st.stop()
                            st.session_state["case"] = {
                                "dwi": dwi,
                                "adc": adc,
                                "gt": gt,
                                "label": "upload",
                                "pred": pred,
                                "method": method,
                            }

    case = st.session_state.get("case")
    if case is None:
        st.info("Selecione um caso na barra lateral e clique em **Carregar e segmentar**.")
        return

    dwi, adc, gt, pred = case["dwi"], case["adc"], case["gt"], case["pred"]
    method_used = case.get("method", "—")
    st.subheader(f"Caso: `{case['label']}` · método: **{method_used}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shape DWI", str(dwi.shape))
    col2.metric("Voxels preditos", f"{int(pred.sum()):,}")
    col3.metric("Volume predito*", f"{volume_ml(pred):.2f} mL")
    if gt is not None and gt.shape == pred.shape:
        col4.metric("Dice", f"{dice_score(pred, gt):.3f}")
        st.caption(
            f"IoU = {iou_score(pred, gt):.3f}  |  "
            "*volume assume espaçamento 1×1×1 mm se não informado."
        )
    else:
        col4.metric("Dice", "—")
        st.caption("*Volume aproximado (espaçamento 1×1×1 mm). GT indisponível ou incompatível.")

    z_default = mid_slice(gt if gt is not None and gt.any() else pred)
    z = st.slider("Fatia (eixo Z)", 0, int(dwi.shape[-1] - 1), z_default)
    show_slice(
        dwi,
        pred,
        gt if gt is not None and gt.shape == pred.shape else None,
        z,
        method_used,
    )

    buf = io.BytesIO()
    np.save(buf, pred.astype(np.uint8))
    st.download_button(
        "Baixar máscara predita (.npy)",
        data=buf.getvalue(),
        file_name=f"{case['label']}_pred_mask.npy",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
