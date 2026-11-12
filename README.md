# Parte prática — TCC (I e II)

Exploração, baseline clássico, **U-Net** e interface CAD no **ISLES-2022**.

## Preparar ambiente

```bash
cd /Users/vitoriakelly/Documents/TCC
source .venv/bin/activate
export MPLCONFIGDIR=./.matplotlib_cache
pip install -r pratica/requirements.txt
```

## TCC I — EDA + baseline

```bash
python pratica/scripts/eda_isles2022.py
python pratica/scripts/baseline_threshold.py
```

## TCC II — U-Net

Smoke test (rápido, poucos casos):

```bash
python pratica/scripts/train_unet.py --epochs 2 --max-subjects 8 --out-name unet2d_smoke
```

Treino completo (40 casos da amostra local):

```bash
python pratica/scripts/train_unet.py --epochs 25 --out-name unet2d_isles
python pratica/scripts/evaluate_unet.py --checkpoint pratica/outputs/checkpoints/unet2d_isles_best.pt
```

Saídas:
- `pratica/outputs/checkpoints/` — pesos `.pt`
- `pratica/outputs/metrics/unet_*.json|csv` — Dice/IoU
- `pratica/outputs/figures/unet_*.png` — curvas e overlays

## Interface CAD

```bash
streamlit run pratica/app/app_streamlit.py
```

Na barra lateral escolha **Baseline (limiar)** ou **U-Net** (exige checkpoint treinado).

## Dataset

Amostra em `ISLES-2022/` (40 casos BIDS: DWI, ADC e máscaras em `derivatives/`).
