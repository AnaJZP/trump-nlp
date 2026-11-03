"""Descriptive and lexical analyses (headlines = webTitle).

Sections, content types, records per day, headline length, body length,
term frequencies, Zipf rank–frequency, bigrams/trigrams (within headlines),
headline TF-IDF (unigrams, min_df=5, max_df=0.95, max 10,000 features) and
named entities (spaCy en_core_web_sm; PERSON, ORG, GPE, NORP).
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import RESULTS, headline_tokens, load_corpus


def ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def main(run_ner=True):
    df = load_corpus()
    title = df["webTitle"].fillna("").astype(str)
    toks = [headline_tokens(t) for t in title]
    freq = Counter(t for ts in toks for t in ts)
    hl = title.str.split().str.len()
    bw = df["bodyText"].str.split().str.len()
    day = pd.to_datetime(df["webPublicationDate"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()

    vec = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.95, max_features=10000)
    X = vec.fit_transform([" ".join(ts) for ts in toks])
    terms = vec.get_feature_names_out()
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()

    out = {
        "n_records": len(df),
        "content_types": df["type"].value_counts().to_dict(),
        "top15_sections": df["sectionName"].value_counts().head(15).reset_index().values.tolist(),
        "daily": {k.strftime("%Y-%m-%d"): int(v) for k, v in day.value_counts().sort_index().items()},
        "headline_len_counts": {int(k): int(v) for k, v in hl.value_counts().sort_index().items()},
        "headline_len": hl.describe().to_dict(),
        "body_words": bw.describe().to_dict(),
        "top25_headline_terms": freq.most_common(25),
        "zipf_freq_of_freq": {int(k): int(v) for k, v in Counter(freq.values()).items()},
        "bigrams_within": Counter(g for ts in toks for g in ngrams(ts, 2)).most_common(20),
        "trigrams_within": Counter(g for ts in toks for g in ngrams(ts, 3)).most_common(20),
        "tfidf_headlines_n_terms": len(terms),
        "tfidf_headlines_top25": [[terms[i], float(mean_tfidf[i])] for i in mean_tfidf.argsort()[::-1][:25]],
        "idf_headlines_lowest10": [[terms[i], float(vec.idf_[i])] for i in vec.idf_.argsort()[:10]],
    }
    if run_ner:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        ents = {k: Counter() for k in ("PERSON", "ORG", "GPE", "NORP")}
        for doc in nlp.pipe(title, batch_size=64):
            for e in doc.ents:
                if e.label_ in ents and len(e.text.strip()) > 1:
                    ents[e.label_][e.text.strip()] += 1
        out["ner_headlines_top10"] = {k: c.most_common(10) for k, c in ents.items()}
    (RESULTS / "descriptive_lexical.json").write_text(json.dumps(out, indent=1, default=float))
    print("Saved results/descriptive_lexical.json")


if __name__ == "__main__":
    main()
