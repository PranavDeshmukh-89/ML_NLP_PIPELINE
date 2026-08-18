# NLP Analytics with ML Models

A production-ready, modular NLP pipeline designed to ingest multi-file Parquet review datasets, perform sarcasm and context-aware sentiment analysis, extract menu items, discover root-cause operational bottlenecks, and generate an executive dashboard exported as an interactive HTML report.

---

## Repository Structure

```text
restaurant-review-nlp/
├── src/
│   ├── ingestion.py
│   ├── roberta.py
│   ├── spacy.py
│   ├── absa.py
│   ├── topic_modeling.py
│   └── plotly.py
├── results/
│   ├── Screenshot 2026-08-18 230743.png
│   ├── Screenshot 2026-08-18 230454.png
│   └── Screenshot 2026-08-18 230535.png
├── requirements.txt
└── README.md                   
```
---

## Executive Dashboard & Visual Analytics

The pipeline automatically compiles model predictions into an interactive Plotly dashboard exported to `restaurant_nlp_report.html`. Below are key visual insights extracted by the pipeline:

### 1. Aspect-Based Sentiment Breakdown
Classifies feedback into core business dimensions (*Food & Drinks*, *Customer Service*, *Value & Price*, *Ambiance & Atmosphere*) using Zero-Shot classification to pinpoint specific operational strengths and weaknesses.

![Aspect-Based Sentiment Breakdown](results/Screenshot%202026-08-18%20230743.png)

### 2. Overall Portfolio Sentiment Distribution
Uses RoBERTa (`twitter-roberta-base-sentiment-latest`) to capture nuanced tone, context, and sarcasm across all customer feedback.

![Overall Sentiment Distribution](results/Screenshot%202026-08-18%20230454.png)

### 3. Item & Menu-Level Specific Mentions
Leverages spaCy POS tagging and noun chunking to surface the top complimented items alongside recurring operational complaints.

![Item & Menu-Level Specific Mentions](results/Screenshot%202026-08-18%20230535.png)

---

## Pipeline Architecture & Key Features

* **Multi-File Parquet Ingestion:** Dynamically detects, validates, and concatenates partitioned Parquet datasets (such as Spark output splits).
* **Target Schema Alignment:** Automatically enforces `reviews_list` as the canonical text column to prevent downstream schema mismatches.
* **Sarcasm & Context-Aware Classification:** Implements transformer-based sentiment analysis capable of detecting complex emotional nuances.
* **Granular Menu Item Extraction:** Groups noun phrases by positive and negative sentiment labels to isolate granular product feedback.
* **Unsupervised Topic Modeling:** Applies TF-IDF vectorization and Non-Negative Matrix Factorization (NMF) strictly on negative review clusters to identify root causes of customer dissatisfaction.

---

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch, Hugging Face `transformers`
* **Natural Language Processing:** `spacy`, `scikit-learn`
* **Data Engineering & IO:** `pandas`, `pyarrow`, `numpy`
* **Interactive Visualization:** `plotly`
