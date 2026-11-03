"""VADER and FinBERT sentiment on the full article body (texto_analisis).

* VADER (NLTK SentimentIntensityAnalyzer) scores the complete text; compound in [-1, 1].
* FinBERT (ProsusAI/finbert) reads the first 5,000 characters, truncated to 512
  tokens; score = P(positive) - P(negative).
* Both are labelled with the ±0.05 thresholds.

A GPU is strongly recommended for FinBERT (a few minutes on a Colab T4).
"""
import json

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import RESULTS, SCORES_CSV, label, load_corpus

ORDER = ["negative", "neutral", "positive"]


def finbert_scores(texts, batch_size=64):
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos = model.config.label2id.get("positive", 0)
    neg = model.config.label2id.get("negative", 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok([t[:5000] for t in texts[i:i + batch_size]], return_tensors="pt",
                      truncation=True, max_length=512, padding=True).to(device)
            p = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
            out.extend((p[:, pos] - p[:, neg]).tolist())
    return np.array(out)


def main():
    df = load_corpus()
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["texto_analisis"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["finbert_score"] = finbert_scores(df["texto_analisis"].tolist())
    df["vader_label"] = df["vader_compound"].apply(label)
    df["finbert_label"] = df["finbert_score"].apply(label)
    df[["id", "vader_compound", "vader_label", "finbert_score", "finbert_label"]].to_csv(SCORES_CSV, index=False)

    rho, p = spearmanr(df["vader_compound"], df["finbert_score"])
    cm = pd.crosstab(df["vader_label"], df["finbert_label"]).reindex(index=ORDER, columns=ORDER, fill_value=0)
    edges = np.linspace(-1, 1, 41)
    summary = {
        "labels": {m: df[f"{m.lower()}_label"].value_counts().reindex(ORDER, fill_value=0).to_dict()
                   for m in ("VADER", "FinBERT")},
        "spearman_rho": float(rho), "p_value": float(p),
        "agreement": float(np.trace(cm.values) / cm.values.sum()),
        "cohen_kappa": float(cohen_kappa_score(df["vader_label"], df["finbert_label"], labels=ORDER)),
        "confusion_matrix_rows_vader_cols_finbert": cm.values.tolist(),
        "hist_vader": np.histogram(df["vader_compound"], bins=edges)[0].tolist(),
        "hist_finbert": np.histogram(df["finbert_score"], bins=edges)[0].tolist(),
        "hist2d": np.histogram2d(df["vader_compound"], df["finbert_score"], bins=[edges, edges])[0].astype(int).tolist(),
    }
    (RESULTS / "sentiment_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("labels", "spearman_rho", "agreement", "cohen_kappa",
                                              "confusion_matrix_rows_vader_cols_finbert")}, indent=1))


if __name__ == "__main__":
    main()
