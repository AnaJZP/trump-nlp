"""Spanish figures used only in the thesis chapter (Figuras 3.1, 3.6 and 3.8)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import FIGURES
OUT = str(FIGURES / "es")
INK, INK_2, GRID = "#1f1f1e", "#52514e", "#e4e3de"
BAR, BAR_2 = "#2a78d6", "#eb6834"
NEG, NEU, POS = "#d03b3b", "#858480", "#2a78d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": "#bdbcb6",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK_2,
    "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "legend.fontsize": 8, "text.color": INK, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print("saved", name)


# Figura 3.1 — flujo de construcción del corpus (vertical)
steps = [
    ("1. The Guardian Content API", "Consulta «Donald Trump», 1 oct 2025 – 25 oct 2026, en bloques trimestrales; campos headline, bodyText y trailText"),
    ("2. Base maestra", "16,747 registros y 12 variables; 0 identificadores y 0 URL duplicados (14,975 article, 1,482 liveblog, 290 interactive)"),
    ("3. Preparación del texto", "Titulares (webTitle) para el análisis léxico; cuerpo completo (bodyText) para sentimiento y tópicos"),
    ("4. Análisis", "Frecuencias, n-gramas y TF-IDF de titulares · VADER y FinBERT · NMF con seis tópicos"),
    ("5. Comparación", "Correlación de Spearman, porcentaje de acuerdo, kappa de Cohen y matriz de confusión"),
]
fig, ax = plt.subplots(figsize=(6.5, 3.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(steps))
ax.axis("off")
for i, (t, d) in enumerate(steps):
    y0 = len(steps) - 1 - i
    ax.add_patch(FancyBboxPatch((0.01, y0 + 0.14), 0.98, 0.66, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#eef4fc", ec=BAR, lw=1.0))
    ax.text(0.035, y0 + 0.47, t, ha="left", va="center", fontsize=7.8, fontweight="bold", color=INK)
    import textwrap
    ax.text(0.40, y0 + 0.47, "\n".join(textwrap.wrap(d, 54)), ha="left", va="center", fontsize=7.2, color=INK_2, linespacing=1.3)
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.15, y0 + 0.02 - 0.0), xytext=(0.15, y0 + 0.14),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0, mutation_scale=9))
save(fig, "fig3_1_flujo_corpus")

# Figura 3.6 — categorías de sentimiento por modelo
cats = ["Negativo", "Neutral", "Positivo"]
vals = {"VADER": [8701, 60, 7986], "FinBERT": [14077, 1575, 1095]}
fig, ax = plt.subplots(figsize=(6.5, 2.6))
y = np.arange(2)
left = np.zeros(2)
cols = [NEG, NEU, POS]
for k, (c, col) in enumerate(zip(cats, cols)):
    v = np.array([vals["VADER"][k], vals["FinBERT"][k]])
    pct = v / 16747
    ax.barh(y, pct, left=left, color=col, height=0.55, edgecolor="white", linewidth=1.5, label=c)
    for yi in range(2):
        if pct[yi] > 0.08:
            ax.text(left[yi] + pct[yi] / 2, yi, f"{pct[yi]:.1%}\n({v[yi]:,})", ha="center", va="center",
                    fontsize=7.5, color="white")
    left += pct
ax.text(0.999, 0.36, "Neutral: 0.4% (60)", ha="right", va="top", fontsize=7, color=INK_2)
ax.text(0.999, 0.66, "Positivo: 6.5% (1,095)", ha="right", va="bottom", fontsize=7, color=INK_2)
ax.set_yticks(y, ["VADER", "FinBERT"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Proporción de registros (N = 16,747)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
save(fig, "fig3_6_categorias_sentimiento")

# Figura 3.8 — registros por tópico dominante
topics = [("T5 Epstein / política partidista de EE. UU.", 4198), ("T0 Genérico / personas y lugares", 3984),
          ("T4 Precios, IA y energía", 3143), ("T3 Inmigración y agentes federales", 2179),
          ("T1 Irán–Israel", 1967), ("T2 Ucrania–Rusia", 1276)]
fig, ax = plt.subplots(figsize=(6.5, 2.8))
yy = np.arange(len(topics))
ax.barh(yy, [n for _, n in topics], color=BAR, height=0.62, edgecolor="white")
for yi, (_, n) in zip(yy, topics):
    ax.text(n + 50, yi, f"{n:,} ({n / 16747:.1%})", va="center", fontsize=7.5, color=INK_2)
ax.set_yticks(yy, [t for t, _ in topics])
ax.invert_yaxis()
ax.set_xlim(0, 5200)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlabel("Número de registros")
ax.grid(axis="y", visible=False)
save(fig, "fig3_8_registros_topico")
