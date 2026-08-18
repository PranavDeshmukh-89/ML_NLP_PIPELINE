# ------------------------------------------------------------------------------
# SECTION 3: ROBERTA SENTIMENT ANALYSIS (CONTEXT & SARCASM AWARE)
# ------------------------------------------------------------------------------
print("\n[3/7] Running RoBERTa sentiment classifier...")
roberta_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
    truncation=True,
    max_length=512
)

results = roberta_sentiment(df[TEXT_COL].astype(str).tolist())
df['roberta_label'] = [res['label'].capitalize() for res in results]
df['roberta_confidence'] = [round(res['score'], 4) for res in results]