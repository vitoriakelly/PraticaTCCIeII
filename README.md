# Parte prática — TCC I e TCC II

Exploração de dados, baseline clássico, U-Net 2D e interface CAD para segmentação de lesão isquêmica em RM (DWI + ADC) no **ISLES-2022**.

## 1. Estrutura esperada

Os caminhos são relativos à **raiz do projeto** (a pasta que contém `pratica/` e `ISLES-2022/`):

```
<raiz do projeto>/
├── ISLES-2022/                 # dados (não incluídos; ver seção 2)
├── .streamlit/config.toml
└── pratica/
    ├── app/app_streamlit.py    # interface CAD
    ├── scripts/                # treino, avaliação, CV, figuras, baseline
    ├── outputs/
    │   ├── checkpoints/        # pesos (.pt) — modelo principal e piloto
    │   ├── metrics/            # CSV/JSON/logs dos experimentos
    │   └── figures/            # figuras usadas no monógrafo
    └── requirements.txt
```

## 2. Dados — ISLES-2022

Coloque a base na raiz do projeto, na pasta `ISLES-2022/`, no formato BIDS:

```
ISLES-2022/
├── sub-strokecase0001/ses-0001/dwi/   sub-strokecase0001_ses-0001_dwi.nii.gz, ..._adc.nii.gz
├── ...
└── derivatives/sub-strokecase0001/ses-0001/   sub-strokecase0001_ses-0001_msk.nii.gz
```

O experimento principal usa **todos os 250 casos** disponíveis. A base pode ser obtida no desafio ISLES-2022 (Petzsche et al., *Scientific Data*, 2022). Os scripts ignoram os demais arquivos (FLAIR, por exemplo).

## 3. Ambiente

```bash
cd <raiz do projeto>
python3 -m venv .venv
source .venv/bin/activate
export MPLCONFIGDIR=./.matplotlib_cache
pip install -r pratica/requirements.txt
```

Testado com Python 3.9 e PyTorch 2.x. O treino usa automaticamente Apple MPS, CUDA ou CPU (`--device auto`). Em CPU, o treino leva bem mais tempo.

## 4. Como reproduzir os resultados

Todos os comandos devem ser executados a partir da raiz do projeto, com o ambiente ativado.

### 4.1 Baseline clássico (TCC I)

```bash
python pratica/scripts/eda_isles2022.py
python pratica/scripts/baseline_threshold.py
```

Saídas: `pratica/outputs/metrics/baseline_*.csv|json`.

### 4.2 Modelo principal — U-Net nos 250 casos (TCC II)

**Opção A — usar o checkpoint já treinado** (`pratica/outputs/checkpoints/unet2d_isles250_best.pt`):

```bash
python pratica/scripts/evaluate_unet.py \
  --checkpoint pratica/outputs/checkpoints/unet2d_isles250_best.pt \
  --postprocess --tag unet250
```

**Opção B — treinar do zero** (cerca de 41 min no Apple MPS):

```bash
python pratica/scripts/train_unet.py --epochs 25 --augment --out-name unet2d_isles250
python pratica/scripts/evaluate_unet.py \
  --checkpoint pratica/outputs/checkpoints/unet2d_isles250_best.pt \
  --postprocess --tag unet250
```

A partição é por sujeito, com semente 42: treino 175, validação 38, teste 37. A avaliação usa apenas o conjunto de teste.

Saídas: `pratica/outputs/metrics/unet250_summary.json` e `unet250_metrics_detailed.csv`; `pratica/outputs/figures/unet250_*.png`.

**Resultado esperado no teste (n = 37), com pós-processamento:**

| Métrica | Valor |
|---|---|
| Dice médio / mediana | 0,521 / 0,664 |
| IoU médio / mediana | 0,412 / 0,497 |
| Precisão / recall médios | 0,794 / 0,513 |
| Tempo médio por caso | ≈ 0,49 s (MPS) |

Pequenas diferenças são esperadas se o treino for refeito em outro hardware.

### 4.3 Piloto com 40 casos (referência do TCC II)

Usado para comparar variantes na mesma amostra do TCC I:

```bash
python pratica/scripts/train_unet.py --epochs 25 --augment --max-subjects 40 --out-name unet2d_isles_aug
python pratica/scripts/evaluate_unet.py \
  --checkpoint pratica/outputs/checkpoints/unet2d_isles_aug_best.pt \
  --postprocess --tag unet_aug
```

### 4.4 Validação cruzada 5-fold (amostra de 40 casos)

```bash
python pratica/scripts/crossval_unet.py --folds 5 --epochs 12 --augment --max-subjects 40 --out-name unet2d_cv
```

Saídas: `pratica/outputs/metrics/unet2d_cv_folds.csv` e `unet2d_cv_summary.json`. A média global reportada é Dice ≈ 0,300.

### 4.5 Figuras para o monógrafo

```bash
python pratica/scripts/generate_cad_figures.py
```

## 5. Interface CAD (Streamlit)

```bash
streamlit run pratica/app/app_streamlit.py
```

Na barra lateral, escolha **Baseline (limiar)** ou **U-Net**. Por padrão, a U-Net carrega `unet2d_isles250_best.pt`. Para isso, o checkpoint precisa existir em `pratica/outputs/checkpoints/`. Em seguida, selecione um caso do ISLES-2022 (ou envie seus próprios NIfTI) e execute a segmentação. A ferramenta é um protótipo de apoio: a decisão final é do especialista.

## 6. Observações

- **Sementes:** todos os scripts usam semente 42 por padrão, o que mantém a mesma partição de sujeitos.
- **Volumes:** são aproximados quando o espaçamento não é lido do cabeçalho NIfTI (assume-se 1×1×1 mm).
- **Checkpoints incluídos:** apenas o modelo principal (`unet2d_isles250_best.pt`) e o piloto de 40 casos (`unet2d_isles_aug_best.pt`). Os demais são artefatos intermediários de teste.
