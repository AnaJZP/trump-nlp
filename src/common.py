"""Shared settings and helpers for the trump-nlp pipeline."""
import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"            # not versioned: full texts are not redistributed
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CORPUS_CSV = DATA / "guardian_trump_corpus.csv"
SCORES_CSV = DATA / "sentiment_scores.csv"
TOPICS_CSV = DATA / "nmf_topics.csv"

QUERY = "Donald Trump"
DATE_FROM, DATE_TO = "2025-10-01", "2026-10-25"
QUARTERS = [("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
            ("2026-07-01", "2026-09-30"), ("2026-10-01", "2026-10-25")]

# Stop-word list used for every headline analysis (word frequencies, n-grams,
# Zipf, headline TF-IDF). It also removes the query terms and reporting verbs.
HEADLINE_STOP = {
    "the", "and", "of", "to", "in", "a", "for", "on", "is", "that", "with", "as", "by", "at", "from", "it",
    "this", "be", "are", "was", "were", "an", "or", "but", "not", "have", "has", "had", "he", "she", "they",
    "we", "you", "his", "her", "their", "its", "our", "your", "who", "what", "which", "when", "where", "how",
    "will", "would", "could", "should", "can", "may", "more", "about", "into", "than", "also", "after",
    "before", "over", "under", "between", "through", "during", "while", "said", "says", "one", "two", "new",
    "news", "trump", "donald"}


def parse_fields(value):
    """The API's `fields` column is stored in the CSV as a Python-dict string."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return json.loads(value)
        except ValueError:
            return {}


def load_corpus():
    """Corpus with the analysis text used for sentiment and NMF:
    bodyText, falling back to headline and then trailText."""
    df = pd.read_csv(CORPUS_CSV, encoding="utf-8-sig", low_memory=False)
    f = df["fields"].apply(parse_fields)
    df["bodyText"] = f.apply(lambda d: str(d.get("bodyText") or ""))
    df["texto_analisis"] = f.apply(
        lambda d: str(d.get("bodyText") or d.get("headline") or d.get("trailText") or ""))
    return df


def headline_tokens(title):
    """Lower-cased alphabetic tokens of 3+ letters, minus HEADLINE_STOP."""
    return [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", str(title).lower()) if t not in HEADLINE_STOP]


def label(score, threshold=0.05):
    """Same thresholds for both models: >= +0.05 positive, <= -0.05 negative."""
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"
