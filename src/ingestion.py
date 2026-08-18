# ------------------------------------------------------------------------------
# SECTION 1: INSTALL DEPENDENCIES & SETUP ENVIRONMENT
# ------------------------------------------------------------------------------
print("[1/7] Installing dependencies and downloading language models...")
!pip install -q transformers torch pyarrow pandas tqdm spacy plotly scikit-learn
!python -m spacy download en_core_web_sm -q

import os
import glob
from collections import Counter
import pandas as pd
import numpy as np
import torch
from google.colab import files
from tqdm import tqdm
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set GPU device
device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"--> Hardware Accelerator: {device_name}")


# ------------------------------------------------------------------------------
# SECTION 2: MULTI-FILE PARQUET INGESTION & TEXT_COL TARGETING
# ------------------------------------------------------------------------------
print("\n[2/7] Loading review dataset(s)...")

# Define target text column explicitly
TEXT_COL = 'reviews_list'

# Detect any pre-existing parquet files in local directory
existing_parquet_files = glob.glob("*.parquet")

# Prompt upload if no local files are found
if not existing_parquet_files:
    print("No existing parquet files found. Please upload your gold layer parquet file(s)...")
    print("--> Note: You can select and upload MULTIPLE files simultaneously.")
    try:
        uploaded = files.upload()
        existing_parquet_files = [f for f in uploaded.keys() if f.endswith('.parquet')]
    except Exception as e:
        print(f"--> Upload interrupted or failed: {e}")

# Fallback: Create mock dataset if no files were uploaded or found
if not existing_parquet_files:
    print("--> Generating realistic sample restaurant review dataset...")
    sample_data = {
        'review_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'reviews_list': [
            "Great, waiting 2 hours for cold pizza is just fantastic!",  # Sarcastic Negative
            "The truffle burger was unbelievable, but the waiter was super rude.",  # Mixed Aspects
            "Extremely overpriced for tiny portions. Never coming back.",  # Price / Food
            "Cozy atmosphere and great music, but the pasta was completely bland.",  # Ambiance / Food
            "Fast service and incredible sushi! Highly recommended.",  # Service / Food
            "Oh marvelous, another wrong delivery order with missing drinks.",  # Sarcastic Negative
            "Garlic bread was warm and crispy. Excellent customer support too.",  # Positive
            "The fish and chips were way too greasy and the bill had extra hidden fees." # Food / Price
        ]
    }
    fallback_file = 'gold_sentiment_sample.parquet'
    df_raw = pd.DataFrame(sample_data)
    df_raw.to_parquet(fallback_file)
    existing_parquet_files = [fallback_file]

# Read all detected parquet files into a list and combine into one DataFrame
dfs = []
for file_path in existing_parquet_files:
    try:
        temp_df = pd.read_parquet(file_path)
        dfs.append(temp_df)
        print(f"--> Successfully loaded {len(temp_df)} records from '{file_path}'.")
    except Exception as e:
        print(f"--> [Warning] Could not read '{file_path}': {e}")

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n--> TOTAL LOADED: {len(df)} reviews across {len(dfs)} file(s).")
else:
    raise ValueError("Error: No valid Parquet files were loaded.")

# Verify target column exists
if TEXT_COL not in df.columns:
    raise KeyError(
        f"Column '{TEXT_COL}' not found in dataset. Available columns: {df.columns.tolist()}"
    )

print(f"--> Target text column assigned: '{TEXT_COL}'.")