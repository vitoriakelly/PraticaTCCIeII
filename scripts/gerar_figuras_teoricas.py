#!/usr/bin/env python3
"""Gera figuras esquemáticas para a fundamentação e a metodologia do TCC I."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "TCC1_extracted" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0,
})


def rounded(ax, xy, w, h, text, fc, ec="#2c3e50", fontsize=10, bold=True):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", color="#1a1a1a",
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.3, color="#2c3e50",
    ))


def fig_hierarquia_ia():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    rounded(ax, (0.6, 0.5), 8.8, 4.2, "", "#e8f1fa", "#3d6ea8")
    ax.text(5.0, 4.3, "Inteligência Artificial (IA)", ha="center", va="center",
            fontsize=13, fontweight="bold")
    rounded(ax, (1.5, 0.9), 7.0, 2.9, "", "#d9efe4", "#2e7d5b")
    ax.text(5.0, 3.45, "Aprendizado de Máquina (ML)", ha="center", va="center",
            fontsize=12, fontweight="bold")
    rounded(ax, (2.5, 1.2), 5.0, 1.7, "Aprendizado Profundo (DL)\nCNNs • U-Net",
            "#fde8d2", "#c45c26", fontsize=11)
    ax.set_title("Hierarquia das subáreas da Inteligência Artificial", pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "hierarquia_ia.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cad():
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.2, "Exame\n(RM/TC)"),
        (2.5, "Pré-\nprocessamento"),
        (4.8, "Segmentação\n/ análise"),
        (7.1, "Quantificação\n(volume)"),
        (9.4, "Validação\ndo especialista"),
    ]
    colors = ["#d6e6f5", "#d9efe4", "#fde8d2", "#efe4d9", "#e8d9ef"]
    for (x, t), c in zip(steps, colors):
        rounded(ax, (x, 0.7), 2.1, 1.6, t, c, fontsize=9)
    for i in range(len(steps) - 1):
        arrow(ax, steps[i][0] + 2.1, 1.5, steps[i + 1][0], 1.5)
    ax.set_title("Fluxo típico de um sistema CAD para apoio ao diagnóstico de AVC", pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_cad.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cnn():
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    blocks = [
        (0.3, 1.1, 1.5, 1.8, "Imagem\nde entrada", "#d6e6f5"),
        (2.2, 0.8, 1.7, 2.4, "Convolução\n+ ReLU", "#d9efe4"),
        (4.3, 1.0, 1.6, 2.0, "Pooling", "#fde8d2"),
        (6.3, 0.7, 1.8, 2.6, "Convolução\n+ ReLU", "#d9efe4"),
        (8.5, 1.15, 1.5, 1.7, "Camada\ndensa", "#efe4d9"),
        (10.3, 1.25, 1.4, 1.5, "Saída", "#e8d9ef"),
    ]
    for x, y, w, h, t, c in blocks:
        rounded(ax, (x, y), w, h, t, c, fontsize=8)
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + blocks[i][2]
        x2 = blocks[i + 1][0]
        arrow(ax, x1, 2.0, x2, 2.0)
    ax.set_title("Arquitetura simplificada de uma Rede Neural Convolucional (CNN)", pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "arquitetura_cnn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_unet():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # encoder (left, descending)
    enc = [(1.1, 5.5, "Encoder\n128"), (1.6, 4.1, "64"), (2.1, 2.7, "32"), (2.6, 1.3, "16")]
    dec = [(7.6, 5.5, "Decoder\n128"), (7.1, 4.1, "64"), (6.6, 2.7, "32"), (6.1, 1.3, "16")]
    for (x, y, t), (x2, y2, t2) in zip(enc, dec):
        rounded(ax, (x, y), 1.8, 0.95, t, "#d6e6f5", fontsize=8)
        rounded(ax, (x2, y2), 1.8, 0.95, t2, "#d9efe4", fontsize=8)
        # skip
        ax.plot([x + 1.8, x2], [y + 0.48, y2 + 0.48], "--", color="#7a5c3a", lw=1.1)
    # bottleneck
    rounded(ax, (4.1, 0.35), 1.8, 0.85, "Gargalo", "#fde8d2", fontsize=9)
    arrow(ax, 3.5, 1.3, 4.2, 1.2)
    arrow(ax, 5.9, 0.78, 6.5, 1.3)
    ax.text(5.0, 6.55, "conexões de atalho (skip)", ha="center", fontsize=9, color="#7a5c3a")
    ax.annotate("", xy=(2.0, 5.45), xytext=(2.5, 2.25),
                arrowprops=dict(arrowstyle="->", color="#3d6ea8"))
    ax.annotate("", xy=(8.0, 2.25), xytext=(8.5, 5.45),
                arrowprops=dict(arrowstyle="->", color="#2e7d5b"))
    ax.text(0.35, 3.5, "contração", rotation=90, va="center", fontsize=9, color="#3d6ea8")
    ax.text(9.55, 3.5, "expansão", rotation=90, va="center", fontsize=9, color="#2e7d5b")
    ax.set_title("Arquitetura U-Net (encoder–decoder com skip connections)", pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "arquitetura_unet.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fluxo_tcc1():
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    items = [
        (8.6, "Revisão bibliográfica\n(AVC, IA, CAD, U-Net)"),
        (7.2, "Seleção da base\nISLES-2022"),
        (5.8, "Análise exploratória\n(EDA, 40 casos)"),
        (4.4, "Baseline clássico\n(limiar DWI + ADC)"),
        (3.0, "Avaliação\n(Dice e IoU)"),
        (1.6, "Pipeline CAD\nprevisto para o TCC II"),
    ]
    colors = ["#d6e6f5", "#d9efe4", "#fde8d2", "#efe4d9", "#e8d9ef", "#f7d6d6"]
    for (y, t), c in zip(items, colors):
        rounded(ax, (2.4, y), 5.2, 1.15, t, c, fontsize=10)
    for i in range(len(items) - 1):
        arrow(ax, 5.0, items[i][0], 5.0, items[i + 1][0] + 1.15)
    ax.set_title("Fluxo metodológico adotado no TCC I", pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fluxo_metodologico_tcc1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_paradigmas_ml():
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    rounded(ax, (0.3, 0.6), 2.9, 1.9, "Supervisionado\nentrada + rótulo\n(ex.: imagem + máscara)", "#d6e6f5", fontsize=9)
    rounded(ax, (3.55, 0.6), 2.9, 1.9, "Não supervisionado\nsem rótulos\n(ex.: agrupamentos)", "#d9efe4", fontsize=9)
    rounded(ax, (6.8, 0.6), 2.9, 1.9, "Por reforço\nrecompensa ao agir\n(não é o foco deste TCC)", "#fde8d2", fontsize=9)
    ax.set_title("Paradigmas clássicos de aprendizado de máquina", pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "paradigmas_ml.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_hierarquia_ia()
    fig_cad()
    fig_cnn()
    fig_unet()
    fig_fluxo_tcc1()
    fig_paradigmas_ml()
    print("Figuras geradas em", OUT)
