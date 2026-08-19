# Parte prática — TCC I

Exploração e baseline no dataset **ISLES-2022**.

## Preparar ambiente

```bash
cd /Users/vitoriakelly/Documents/TCC
source .venv/bin/activate
export MPLCONFIGDIR=./.matplotlib_cache
```

## Rodar

```bash
python pratica/scripts/eda_isles2022.py
python pratica/scripts/baseline_threshold.py
```

## Saídas

- `pratica/outputs/figures/` — figuras geradas
- `pratica/outputs/metrics/` — CSV/JSON com estatísticas e Dice/IoU
- Cópias das figuras também em `TCC1_extracted/figuras/` para o LaTeX

## Dataset

Amostra de 40 casos extraída de `ISLES-2022.zip`. 
