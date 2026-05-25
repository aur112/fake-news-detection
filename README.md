# Fake News Detection — Linguistic Analysis & Stacking Classifier

A two-part NLP project exploring whether surface linguistic features can distinguish fake news from real news, built for SDSC2005.

---

## Overview

**Research question:** Can linguistic cues (word choice, sentiment, writing style) reliably separate fake news from real news?

**Subtask 1** — Descriptive analysis on FakeNewsNet: computes interpretable features (article length, type–token ratio, VADER sentiment, punctuation patterns) and tests whether they differ statistically between fake and real articles.

**Subtask 2** — Supervised classification on a merged corpus (~52k articles from FakeNewsNet + LIAR + ISOT): trains four base learners across three feature views (word TF-IDF, character TF-IDF, sentence embeddings), then combines them with a stacked meta-learner.

---

## Results

| Model | Macro F1 | ROC-AUC |
|---|---|---|
| LR + Word TF-IDF | ~0.90 | — |
| SVM + Char TF-IDF | ~0.91 | — |
| Random Forest + Word TF-IDF | ~0.91 | — |
| LR + Sentence Embeddings | ~0.85 | — |
| **Stacked Ensemble (meta-LR)** | **~0.91** | **~0.98** |

Evaluated on a stratified 20% held-out test split (`random_state=42`).

---

## Datasets

| Dataset | Source | Size | Labels |
|---|---|---|---|
| FakeNewsNet | [Kaggle](https://www.kaggle.com/datasets/mdepak/fakenewsnet) | ~422 articles | PolitiFact + BuzzFeed |
| LIAR | [Kaggle](https://www.kaggle.com/datasets/csmalarkodi/liar-fake-news-dataset) | ~12k claims | 6-class → binarized |
| ISOT | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | ~39k articles | fake / real |

All three are downloaded automatically via `kagglehub` on first run.

---

## Setup

**Requirements:** Python 3.8+

```bash
pip install kagglehub pandas numpy matplotlib seaborn scipy nltk scikit-learn sentence-transformers
```

You'll also need a Kaggle account with an API token configured (`~/.kaggle/kaggle.json`) for `kagglehub` to download the datasets.

---

## Usage

Open and run `fakenewsdetection_main.ipynb` end-to-end. Cells are ordered sequentially — run all from top to bottom. The first run will download the datasets (~500MB total); subsequent runs use the cached versions.

---

## Project Structure

```
fakenewsdetection_main.ipynb   # Main notebook (EDA + classifier)
README.md
```

---

## Key Findings

- **Sentiment is a source-specific signal**: BuzzFeed fake articles skew strongly negative on VADER; PolitiFact fake/real articles are similar in sentiment.
- **Length features overlap heavily**: Word count alone is not a reliable separator between fake and real.
- **Stacking helps marginally** over the best single base model, with the calibrated SVM on character n-grams being the strongest individual learner.
- **Embeddings underperform** bag-of-words models on this corpus, likely because stylistic cues matter more than semantics here.

---

## Limitations

- High accuracy partly reflects **dataset-specific patterns** (e.g., ISOT has known stylistic artifacts) rather than human-grade fact-checking ability.
- The model has no access to **source credibility, images, URLs, or social context** — all strong real-world signals.
- Evaluation uses a **single train/test split**; nested cross-validation would give stricter estimates.
- LIAR (short political claims) and ISOT (full news articles) differ in **genre and length** from FakeNewsNet, so the merged corpus is not domain-homogeneous.

---

## Acknowledgements

Datasets: FakeNewsNet ([arXiv:1809.01286](https://arxiv.org/abs/1809.01286)), LIAR, ISOT. Sentence embeddings via `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/).
