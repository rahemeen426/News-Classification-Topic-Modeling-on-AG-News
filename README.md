# News Classification & Topic Modeling on AG News

Supervised text classification and unsupervised topic discovery on a 127K-article news corpus — comparing a TF-IDF SVM baseline, a rule-based hybrid system, and a fine-tuned BERT transformer for classification, and LDA vs. NMF for topic modeling.

<!-- Optional badges — uncomment / edit once the repo is live
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
-->

## Overview

Online news volume is growing exponentially, and manual categorization and content analysis is time-consuming, subjective, and doesn't scale. This project tackles the problem from two complementary angles:

- **Supervised classification** — categorize articles into four predefined topics (World, Sports, Business, Sci/Tech) using SVM, a hybrid rule+SVM system, and a fine-tuned BERT transformer.
- **Unsupervised topic modeling** — discover latent thematic structure in the corpus using LDA and NMF, evaluating coherence, interpretability, and computational efficiency.

## Dataset

**[AG News](https://huggingface.co/datasets/sh0416/ag_news)** (via Hugging Face Datasets)

| | |
|---|---|
| Total articles | 127,600 |
| Training samples | 120,000 |
| Test samples | 7,600 |
| Categories | 4 — World (0), Sports (1), Business (2), Sci/Tech (3) |

- Perfect class balance: each category is exactly 25% of the corpus, eliminating imbalance-driven classification bias.
- Each sample is the concatenated article title + description.

## Pipeline

The end-to-end workflow covers data loading & preprocessing → EDA → supervised classification → topic modeling (LDA + NMF) → comparative evaluation.

```
Data Loading & Preprocessing
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ├──► Supervised Classification (SVM · Hybrid · BERT) ──► Comparative Evaluation
        │
        └──► Topic Modeling (LDA · NMF) ──► Comparative Evaluation
```

*(Add the actual pipeline diagram from the presentation, e.g. `assets/architecture_workflow.png`, once the project files are uploaded.)*

## Exploratory Data Analysis

- **World** news has the highest average length (38.88 words / 242.63 characters).
- **Sci/Tech** shows the greatest variability (word variance 154.20, character variance 6763.27) — ranging from brief software updates to comprehensive reports, and dominates the longest-article outliers (36.27% of all outliers, top 5 longest articles).
- **Sports** has the lowest average character count (224.65) despite moderate word counts.
- **Business** has the most consistent article lengths (only 10.17% of outliers) but represents 80% of the shortest articles (5–8 words).
- 3,568 outlier articles identified across all categories.

## Supervised Classification Models

| # | Model | Approach |
|---|---|---|
| 1 | **SVM + TF-IDF** *(baseline)* | TF-IDF vectorization (max_features=10,000, stop-word removal) + linear SVM |
| 2 | **Hybrid System** | Two-stage decision: keyword rule matching (curated domain keywords) with fallback to SVM |
| 3 | **BERT (Transformer)** *(state-of-the-art)* | Fine-tuned BERT-base — 12 encoder layers, 12 attention heads, 768-dim hidden representation; multi-head self-attention for contextual/long-range dependencies. LR = 2×10⁻⁵, batch size = 8 |

### Results

| Model | Accuracy | F1 |
|---|---|---|
| SVM + TF-IDF | 91.00% | 0.9098 |
| Hybrid (Rule + SVM) | 88.70% | 0.8863 |
| **BERT** | **94.50%** | **0.9450** |

### Case study — why context matters

For the headline *"Apple Stock"*:

| Model | Prediction |
|---|---|
| SVM | Sci/Tech |
| Hybrid | Sci/Tech |
| **BERT** | **Business** ✓ |

BERT's attention mechanism weighs "stock" more heavily than "Apple," and its contextual embeddings recognize the financial-equity context — resolving an ambiguity that keyword- and frequency-based models miss.

## Topic Modeling: LDA vs. NMF

| | LDA (Latent Dirichlet Allocation) | NMF (Non-Negative Matrix Factorization) |
|---|---|---|
| Type | Probabilistic generative model | Linear algebraic decomposition — V ≈ W × H |
| Vectorization | CountVectorizer (word frequencies) | TF-IDF (discriminative term weighting) |
| Learning | Online learning for efficiency | Non-negativity → additive, interpretable topics |
| Config | max_df=0.95, min_df=2, max_features=2000 | Sharper, more distinctive topics |

**LDA results:**

| Topic | Count | % |
|---|---|---|
| Oil & Energy Prices | 609 | 30.45% |
| Video Games & Entertainment | 483 | 24.15% |
| US Politics & Elections | 343 | 17.15% |
| International Relations | 324 | 16.20% |
| Theme: Make — India — New | 241 | 12.05% |

**NMF results:**

| Topic | Count | % |
|---|---|---|
| Software & Operating Systems | 664 | 33.20% |
| Middle East Conflict | 559 | 27.95% |
| Video Games & Entertainment | 524 | 26.20% |
| Oil & Energy Prices | 145 | 7.25% |
| US Politics & Elections | 108 | 5.40% |

**Key differences:** LDA's largest topic (Oil & Energy, 30.45%) is NMF's smallest (7.25%); NMF's dominant topic (Software & OS, 33.2%) is LDA's smallest (12.05%). LDA leans toward energy/political discourse, NMF toward technology/entertainment.

### Computational performance & topic quality

- NMF was **~10.6× faster** than LDA (7.783s time reduction) — more efficient for large datasets and iterative modeling.
- **Topic diversity** = proportion of unique words among top keywords across topics; NMF scored higher, indicating distinct themes with minimal overlap.
- **Topic purity** = alignment between discovered topics and true document categories; higher purity = better separation.
- pyLDAvis visualization: NMF's topic bubbles are more separated (cleaner boundaries) than LDA's, which shows moderate overlap. LDA's top topic mixes business/energy/tech terms, while NMF's top topic is purely sports-related — reflecting NMF's sharper topic boundaries.

## Supervised vs. Unsupervised: When to Use Which

| Aspect | Topic Modeling | Supervised Models |
|---|---|---|
| Input | Raw text only | Text + pre-defined labels |
| Goal | Discover latent themes | Predict known categories |
| Training | Unsupervised (no labels) | Supervised (labeled data) |
| Output | Topic distributions, keywords | Class predictions (0–3) |
| Validation | Coherence, manual inspection | Accuracy, Precision, Recall, F1 |
| Application | Exploratory analysis | Automated categorization |

## Key Takeaways

- **SVM + TF-IDF** is an effective, fast baseline at 91% accuracy.
- **BERT** achieves the best accuracy (94.5%) and resolves contextual ambiguities that simpler models miss.
- **Hybrid Rule-SVM** offers high interpretability and precision on keyword-rich articles.
- **NMF outperforms LDA** on this corpus: 10.6× faster, semantically tighter topics, better category separation.
- Most classification errors stem from dataset ambiguity — short articles and category overlap.

## Future Work

- Multi-label classification to handle category overlap and boundary cases
- Automated keyword extraction to strengthen the hybrid system
- Real-time and multilingual news analysis to test scalability and robustness
- Semi-supervised classification using topic modeling insights for richer, hierarchical labels

## Tech Stack

`Python` · `scikit-learn` (SVM, TF-IDF, LDA, NMF) · `Transformers` (BERT) · `pandas` / `numpy` · `matplotlib` / `seaborn` · `pyLDAvis`

## Project Structure

```
├── data/                  # AG News dataset (raw / processed)
├── notebooks/             # EDA, modeling, and evaluation notebooks
├── src/                   # Preprocessing, training, and evaluation scripts
├── models/                # Saved model artifacts
├── reports/                # Figures, dashboards, pyLDAvis outputs
└── README.md
```

# AG News Intelligence Hub (How to Run)

## Installation

1.  Ensure you have Python installed (3.8+ recommended).
2.  Install dependencies:
    ```bash
    py -m pip install -r requirements.txt
    ```

## Execution

Run the Streamlit app:
```bash
py -m streamlit run app.py
```

## Note
- The first run will download the AG News dataset and the BERT model, which may take a minute.
- SVM model is trained on the fly on first load and cached.


## Author

**Rahemeen Mukhtiar** — MS Data Science, Università degli Studi di Milano-Bicocca
