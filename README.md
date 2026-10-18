# Parte prática — TCC I

Exploração, baseline e interface CAD (prova de conceito) no **ISLES-2022**.

## Preparar ambiente

```bash
cd /Users/vitoriakelly/Documents/TCC
source .venv/bin/activate
export MPLCONFIGDIR=./.matplotlib_cache
pip install -r pratica/requirements.txt
```

## Rodar análise (EDA + baseline)

```bash
python pratica/scripts/eda_isles2022.py
python pratica/scripts/baseline_threshold.py
```

## Rodar interface CAD

```bash
streamlit run pratica/app/app_streamlit.py
```

Abre no navegador (em geral `http://localhost:8501`).
Na barra lateral: escolha um caso do ISLES-2022 ou faça upload de DWI/ADC.

## Saídas

- `pratica/outputs/figures/` — figuras geradas
- `pratica/outputs/metrics/` — CSV/JSON com estatísticas e Dice/IoU

## Dataset

Amostra de 40 casos em `ISLES-2022/`.
