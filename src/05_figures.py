"""Draws the 11 figures of the article from the JSON files in results/.

    python src/05_figures.py          # English (article)
    python src/05_figures.py --lang es  # Spanish (thesis)

It prefers results produced by scripts 02–04 (sentiment_summary.json,
nmf_topics.json, descriptive_lexical.json) and otherwise uses the shipped
results (notebook_outputs.json and rerun_aggregates_2026-11-02.json).
PNG at 600 dpi and vector PDF are written to figures/<lang>/.
"""
import argparse
import json
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from common import FIGURES, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--lang", choices=["en", "es"], default="en")
LANG = ap.parse_args().lang

T = {  # all visible text, by language
    "en": dict(records="Number of records", mentions="Mentions in headlines", freq_head="Frequency in headlines",
               freq="Frequency", bigr="A. Bigrams", trig="B. Trigrams", tfidf_a="A. Highest mean TF-IDF (top 25)",
               tfidf_x="Mean TF-IDF weight", idf_b="B. Lowest IDF (most widespread)", idf_x="IDF",
               neg="Negative", neu="Neutral", pos="Positive", fin_lab="FinBERT label", vad_lab="VADER label",
               rowshare="Share of VADER row", vmean="VADER (mean compound)", fmean="FinBERT (mean P+ − P−)",
               meanscore="Mean sentiment score", hl="Headline length (words)", nhl="Number of headlines",
               mean="Mean", median="Median", perday="Records per day", daily="Daily count",
               roll="7-day rolling mean", obs="Observed", zipf="Zipf reference (slope = −1)",
               ols="OLS fit, ranks 10–1,000 (slope = {s})", rank="Rank (log scale)", fl="Frequency (log scale)",
               dist="A. Score distributions", logrec="Number of records (log scale)", score="Sentiment score",
               vs="B. VADER vs FinBERT", vc="VADER compound", fs="FinBERT score", reclog="Records (log scale)",
               vleg="VADER (compound)", fleg="FinBERT (P+ − P−)",
               topics={"T0": "T0 Generic / people and places", "T1": "T1 Iran–Israel", "T2": "T2 Ukraine–Russia",
                       "T3": "T3 Immigration enforcement", "T4": "T4 Prices, AI and energy",
                       "T5": "T5 Epstein / US party politics"},
               sections={}, months=None),
    "es": dict(records="Número de registros", mentions="Menciones en titulares", freq_head="Frecuencia en titulares",
               freq="Frecuencia", bigr="A. Bigramas", trig="B. Trigramas", tfidf_a="A. Mayor TF-IDF promedio (25)",
               tfidf_x="Peso TF-IDF promedio", idf_b="B. Menor IDF (más extendidos)", idf_x="IDF",
               neg="Negativo", neu="Neutral", pos="Positivo", fin_lab="Etiqueta FinBERT", vad_lab="Etiqueta VADER",
               rowshare="Proporción de la fila VADER", vmean="VADER (compound medio)", fmean="FinBERT (P+ − P− medio)",
               meanscore="Puntuación media de sentimiento", hl="Longitud del titular (palabras)",
               nhl="Número de titulares", mean="Media", median="Mediana", perday="Registros por día",
               daily="Registros diarios", roll="Media móvil de 7 días", obs="Observado",
               zipf="Referencia de Zipf (pendiente = −1)", ols="Ajuste MCO, rangos 10–1,000 (pendiente = {s})",
               rank="Rango (escala log)", fl="Frecuencia (escala log)", dist="A. Distribución de puntuaciones",
               logrec="Número de registros (escala log)", score="Puntuación de sentimiento", vs="B. VADER frente a FinBERT",
               vc="VADER (compound)", fs="Puntuación FinBERT", reclog="Registros (escala log)",
               vleg="VADER (compound)", fleg="FinBERT (P+ − P−)",
               topics={"T0": "T0 Genérico / personas y lugares", "T1": "T1 Irán–Israel", "T2": "T2 Ucrania–Rusia",
                       "T3": "T3 Inmigración y agentes federales", "T4": "T4 Precios, IA y energía",
                       "T5": "T5 Epstein / política partidista de EE. UU."},
               sections={"US news": "US news (EE. UU.)", "World news": "World news (internacional)",
                         "Opinion": "Opinion (opinión)", "Business": "Business (negocios)",
                         "Australia news": "Australia news", "Politics": "Politics (política)",
                         "Football": "Football (fútbol)", "Sport": "Sport (deportes)",
                         "Technology": "Technology (tecnología)", "Environment": "Environment (medio ambiente)",
                         "Media": "Media (medios)", "Film": "Film (cine)", "Culture": "Culture (cultura)",
                         "News": "News (noticias)", "UK news": "UK news (Reino Unido)"},
               months=["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]),
}[LANG]
OUT = FIGURES / LANG
OUT.mkdir(parents=True, exist_ok=True)


def num(v, d=0):
    """Thousands with comma and decimal point (both languages, as in the manuscripts)."""
    return f"{v:,.{d}f}"


# ---------------------------------------------------------------- style
INK, INK_2, GRID = "#1f1f1e", "#52514e", "#e4e3de"
BAR, BAR_2 = "#2a78d6", "#eb6834"
POS = "#2a78d6"
FULL_W = 6.5
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": "#bdbcb6",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK_2,
    "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "legend.fontsize": 8, "text.color": INK, "figure.dpi": 110, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("saved", OUT / name)


def hbar(ax, labels, values, color=BAR, fmt=lambda v: num(v)):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.66, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    vmax = max(values)
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt(v), va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, vmax * 1.12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: num(v)))


def load(name, fallback):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else fallback


NB = json.loads((RESULTS / "notebook_outputs.json").read_text())
RR = json.loads((RESULTS / "rerun_aggregates_2026-11-02.json").read_text())
LEX = load("descriptive_lexical.json", {**NB, **RR})
SENT = load("sentiment_summary.json", {"confusion_matrix_rows_vader_cols_finbert": RR["cm"], "spearman_rho": RR["rho"],
                                       "hist_vader": RR["hist_vader"], "hist_finbert": RR["hist_finbert"],
                                       "hist2d": RR["hist2d"]})
TOP = load("nmf_topics.json", NB["nmf_topics"])

# Figure 1 — sections
sec = LEX["top15_sections"]
fig, ax = plt.subplots(figsize=(FULL_W, 4.0))
hbar(ax, [T["sections"].get(s, s) for s, _ in sec], [c for _, c in sec])
ax.set_xlabel(T["records"])
save(fig, "fig1_sections")

# Figure 2 — headline length
hc = {int(k): v for k, v in LEX["headline_len_counts"].items()}
xs = np.arange(min(hc), max(hc) + 1)
ys = np.array([hc.get(x, 0) for x in xs])
st = LEX["headline_len"]
fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
ax.bar(xs, ys, width=0.9, color=BAR, edgecolor="white", linewidth=0.6)
ax.axvline(st["mean"], color=INK, lw=1, ls="--")
ax.axvline(st["50%"], color=INK_2, lw=1, ls=":")
top = ys.max() * 1.08
ax.set_ylim(0, top * 1.05)
ax.text(st["mean"] + 0.4, top * 0.97, f"{T['mean']} = {st['mean']:.2f}", fontsize=7.5)
ax.text(st["50%"] - 0.4, top * 0.97, f"{T['median']} = {st['50%']:.0f}", fontsize=7.5, ha="right", color=INK_2)
ax.set_xlabel(T["hl"])
ax.set_ylabel(T["nhl"])
ax.set_xticks(np.arange(5, xs.max() + 1, 5))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: num(v)))
save(fig, "fig2_headline_length")

# Figure 3 — records per day
daily = pd.Series(LEX["daily"])
daily.index = pd.to_datetime(daily.index)
daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
ax.bar(daily.index, daily.values, width=1.0, color="#b7d3f6", label=T["daily"])
ax.plot(daily.index, daily.rolling(7, center=True).mean(), color=BAR, lw=1.6, label=T["roll"])
ax.set_ylabel(T["perday"])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.grid(axis="x", visible=False)
ax.set_xlim(daily.index.min() - pd.Timedelta(days=2), daily.index.max() + pd.Timedelta(days=2))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
if T["months"]:
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{T['months'][mdates.num2date(x).month - 1]}\n{mdates.num2date(x).year}"))
else:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
save(fig, "fig3_daily_volume")

# Figure 4 — named entities
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.6))
for ax, (k, ttl) in zip(axes.ravel(), [("PERSON", "A. PERSON"), ("ORG", "B. ORG"), ("GPE", "C. GPE"), ("NORP", "D. NORP")]):
    rows = LEX["ner_headlines_top10"][k]
    hbar(ax, [r[0] for r in rows], [r[1] for r in rows])
    ax.set_title(ttl)
    ax.tick_params(axis="y", labelsize=7.5)
for ax in axes[1]:
    ax.set_xlabel(T["mentions"])
fig.tight_layout(h_pad=1.6, w_pad=2.2)
save(fig, "fig4_named_entities")

# Figure 5 — top 25 headline terms
t25 = LEX["top25_headline_terms"]
fig, ax = plt.subplots(figsize=(FULL_W, 5.0))
hbar(ax, [w for w, _ in t25], [c for _, c in t25])
ax.set_xlabel(T["freq_head"])
save(fig, "fig5_top_terms")

# Figure 6 — Zipf
fof = {int(k): v for k, v in LEX["zipf_freq_of_freq"].items()}
freqs = np.array(sorted([f for f, n in fof.items() for _ in range(n)], reverse=True), dtype=float)
ranks = np.arange(1, len(freqs) + 1)
m = (ranks >= 10) & (ranks <= 1000)
slope, icpt = np.polyfit(np.log10(ranks[m]), np.log10(freqs[m]), 1)
fig, ax = plt.subplots(figsize=(FULL_W, 3.2))
ax.loglog(ranks, freqs, color=BAR, lw=1.6, label=T["obs"])
ax.loglog(ranks, freqs[0] / ranks, color=INK_2, lw=1, ls="--", label=T["zipf"])
ax.loglog(ranks[m], 10 ** (icpt + slope * np.log10(ranks[m])), color=BAR_2, lw=1.4,
          label=T["ols"].format(s=f"{slope:.2f}".replace("-", "−")))
ax.set_xlabel(T["rank"])
ax.set_ylabel(T["fl"])
ax.legend(loc="lower left")
save(fig, "fig6_zipf")

# Figure 7 — n-grams within headlines
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 5.0))
for ax, key, ttl, col in ((axes[0], "bigrams_within", T["bigr"], BAR), (axes[1], "trigrams_within", T["trig"], BAR_2)):
    rows = LEX[key]
    hbar(ax, [g for g, _ in rows], [c for _, c in rows], color=col)
    ax.set_title(ttl)
    ax.set_xlabel(T["freq"])
    ax.tick_params(axis="y", labelsize=7.5)
fig.tight_layout(w_pad=2)
save(fig, "fig7_ngrams")

# Figure 8 — headline TF-IDF and IDF
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
tf, idf = LEX["tfidf_headlines_top25"], LEX["idf_headlines_lowest10"]
hbar(axes[0], [t for t, _ in tf], [v for _, v in tf], fmt=lambda v: f"{v:.4f}")
axes[0].set_title(T["tfidf_a"])
axes[0].set_xlabel(T["tfidf_x"])
axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
axes[0].tick_params(axis="y", labelsize=7.5)
hbar(axes[1], [t for t, _ in idf], [v for _, v in idf], color=BAR_2, fmt=lambda v: f"{v:.2f}")
axes[1].set_title(T["idf_b"])
axes[1].set_xlabel(T["idf_x"])
axes[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
fig.tight_layout(w_pad=2)
save(fig, "fig8_tfidf_idf")

# Figure 9 — score distributions and joint density
edges = np.linspace(-1, 1, 41)
H = np.array(SENT["hist2d"], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
axes[0].stairs(SENT["hist_vader"], edges, color=POS, lw=1.6, label=T["vleg"])
axes[0].stairs(SENT["hist_finbert"], edges, color=BAR_2, lw=1.6, label=T["fleg"])
for x in (-0.05, 0.05):
    axes[0].axvline(x, color=INK_2, lw=0.7, ls=":")
axes[0].set_xlim(-1, 1)
axes[0].set_yscale("log")
axes[0].set_ylim(5, 20000)
axes[0].set_title(T["dist"])
axes[0].set_xlabel(T["score"])
axes[0].set_ylabel(T["logrec"])
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: num(x)))
axes[0].legend(loc="lower center")
pcm = axes[1].pcolormesh(edges, edges, np.ma.masked_where(H == 0, H).T, cmap="Blues", norm=LogNorm(vmin=1, vmax=H.max()))
axes[1].set_title(T["vs"])
axes[1].set_xlabel(T["vc"])
axes[1].set_ylabel(T["fs"])
axes[1].grid(False)
axes[1].text(-0.95, 0.9, f"Spearman ρ = {SENT['spearman_rho']:.3f}\np < .001", fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
cb = fig.colorbar(pcm, ax=axes[1], fraction=0.046, pad=0.03)
cb.set_label(T["reclog"], fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.tight_layout(w_pad=1.5)
save(fig, "fig9_score_distributions")

# Figure 10 — confusion matrix
cm = np.array(SENT["confusion_matrix_rows_vader_cols_finbert"])
order = [T["neg"], T["neu"], T["pos"]]
row = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3), order)
ax.set_yticks(range(3), order)
ax.set_xlabel(T["fin_lab"])
ax.set_ylabel(T["vad_lab"])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
for i in range(3):
    for j in range(3):
        pct = f"{row[i, j]:.1%}"
        ax.text(j, i, f"{num(cm[i, j])}\n({pct})", ha="center", va="center", fontsize=8,
                color="white" if row[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.text(2.62, i, f"n = {num(cm[i].sum())}", va="center", fontsize=7.5, color=INK_2)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.22, format=mtick.PercentFormatter(1.0))
cb.set_label(T["rowshare"], fontsize=7.5)
cb.ax.tick_params(labelsize=7)
save(fig, "fig10_confusion_matrix")

# Figure 11 — mean sentiment by topic
tp = sorted(TOP, key=lambda t: t["vader_mean"])
fig, ax = plt.subplots(figsize=(FULL_W, 3.6))
y = np.arange(len(tp))
h = 0.36
ax.barh(y - h / 2, [t["vader_mean"] for t in tp], height=h, color=POS, label=T["vmean"], edgecolor="white", linewidth=0.6)
ax.barh(y + h / 2, [t["finbert_mean"] for t in tp], height=h, color=BAR_2, label=T["fmean"], edgecolor="white", linewidth=0.6)
for yi, t in zip(y, tp):
    for off, v in ((-h / 2, t["vader_mean"]), (h / 2, t["finbert_mean"])):
        ax.text(v + (0.015 if v >= 0 else -0.015), yi + off, f"{v:+.2f}".replace("-", "−"), va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK_2)
ax.set_yticks(y, [f"{T['topics'][t['topic']]} (n = {num(t['n'])})" for t in tp])
ax.invert_yaxis()
ax.axvline(0, color=INK_2, lw=0.8)
ax.set_xlim(-0.8, 0.5)
ax.set_xlabel(T["meanscore"])
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig11_sentiment_by_topic")
