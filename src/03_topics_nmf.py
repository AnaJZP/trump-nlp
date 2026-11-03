"""Exploratory NMF topics on the article bodies, and mean sentiment by topic.

TF-IDF: 5,000 features, min_df=5, max_df=0.95, unigrams and bigrams, English
stop words. NMF: 6 components, nndsvda initialisation, 300 iterations, seed 42.
Each record is assigned the component with the highest weight.
Run after 02_sentiment.py if you want the mean sentiment by topic.
"""
import json

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from common import RESULTS, SCORES_CSV, TOPICS_CSV, load_corpus


def main():
    df = load_corpus()
    vec = TfidfVectorizer(stop_words="english", max_features=5000, min_df=5, max_df=0.95, ngram_range=(1, 2))
    X = vec.fit_transform(df["texto_analisis"])
    nmf = NMF(n_components=6, random_state=42, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    terms = vec.get_feature_names_out()
    df["topic"] = W.argmax(axis=1)
    topics = []
    for k, comp in enumerate(nmf.components_):
        topics.append({"topic": f"T{k}", "terms": [terms[i] for i in comp.argsort()[-5:][::-1]],
                       "n": int((df["topic"] == k).sum())})
    df[["id", "topic"]].to_csv(TOPICS_CSV, index=False)
    if SCORES_CSV.exists():
        s = df[["id", "topic"]].merge(pd.read_csv(SCORES_CSV), on="id")
        means = s.groupby("topic")[["vader_compound", "finbert_score"]].mean()
        for t in topics:
            k = int(t["topic"][1:])
            t["vader_mean"] = float(means.loc[k, "vader_compound"])
            t["finbert_mean"] = float(means.loc[k, "finbert_score"])
    (RESULTS / "nmf_topics.json").write_text(json.dumps(topics, indent=1))
    for t in topics:
        print(t)


if __name__ == "__main__":
    main()
