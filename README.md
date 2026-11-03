# trump-nlp

Code, aggregated results and figures for the article:

> Carrillo Ramírez, V., Jiménez Preciado, A. L., & Ríos Bolívar, H. *Sentiment Analysis of The Guardian’s Coverage of Donald Trump: Comparing VADER, FinBERT, and Topic Modelling on News Texts.* (Manuscript submitted for publication.)

The study analyses 16,747 items published by *The Guardian* between 1 October 2025 and 25 October 2026 and retrieved from the Content API with the query “Donald Trump”. It combines descriptive corpus statistics, a lexical analysis of headlines, TF-IDF, exploratory topic modelling with NMF, and a comparison of two sentiment tools: VADER (lexicon and rules) and FinBERT (a transformer fine-tuned on financial text).

## Main results

| | Negative | Neutral | Positive |
|---|---|---|---|
| VADER | 8,701 (51.96%) | 60 (0.36%) | 7,986 (47.69%) |
| FinBERT | 14,077 (84.06%) | 1,575 (9.40%) | 1,095 (6.54%) |

Spearman’s ρ = 0.396 (*p* < .001); categorical agreement = 54.74%; Cohen’s κ = 0.149.
These figures were reproduced exactly by an independent re-run on 2 November 2026 (Colab, T4 GPU).

## What is used for what

| Analysis | Text | Settings |
|---|---|---|
| VADER | Full article body (`bodyText`) | NLTK `SentimentIntensityAnalyzer`, no truncation; ±0.05 thresholds |
| FinBERT | Full article body | `ProsusAI/finbert`; first 5,000 characters, 512 tokens; score = P(pos) − P(neg); ±0.05 thresholds |
| NMF topics | Full article body | TF-IDF (5,000 features, min_df 5, max_df 0.95, 1–2-grams, English stop words); 6 components, nndsvda, 300 iterations, seed 42 |
| Word frequencies, Zipf, n-grams | Headlines (`webTitle`) | Alphabetic tokens of 3+ letters; custom stop words incl. *trump*, *donald*, *said*, *says*; n-grams counted within each headline |
| Headline TF-IDF | Headlines | Unigrams, min_df 5, max_df 0.95, ≤ 10,000 features (5,072 retained) |
| Named entities | Headlines | spaCy `en_core_web_sm`; PERSON, ORG, GPE, NORP |

## Repository layout

```
src/
  common.py               shared settings (query, period, stop words, text variable)
  01_download_guardian.py retrieves the corpus from the Content API (quarterly blocks)
  02_sentiment.py         VADER + FinBERT scores and agreement statistics
  03_topics_nmf.py        NMF topics and mean sentiment by topic
  04_headlines.py         descriptive statistics, lexical analyses and NER
  05_figures.py           the 11 figures of the article (--lang en | es)
results/                  aggregated outputs (no article text)
figures/en, figures/es    figures at 600 dpi (PNG) and vector PDF
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GUARDIAN_API_KEY=your_key          # free developer key: https://open-platform.theguardian.com/
cd src
python 01_download_guardian.py            # writes data/guardian_trump_corpus.csv
python 02_sentiment.py                    # a GPU is recommended for FinBERT
python 03_topics_nmf.py
python 04_headlines.py
python 05_figures.py --lang en
```

`05_figures.py` also runs on its own: without the outputs of steps 02–04 it draws the figures from the shipped files in `results/`.

A later retrieval may return a slightly different set of items if *The Guardian* has added, removed or re-tagged content since October 2026.

## Data

The full texts are not redistributed, in keeping with the terms of use of The Guardian Content API. The `results/` folder holds only aggregated outputs: counts, frequencies, histograms and summary statistics. Never commit an API key: the scripts read it from the `GUARDIAN_API_KEY` environment variable.

## Authors

Víctor Carrillo Ramírez, Ana Lorena Jiménez Preciado and Humberto Ríos Bolívar.
