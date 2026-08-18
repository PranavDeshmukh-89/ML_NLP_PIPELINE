# ------------------------------------------------------------------------------
# SECTION 5: ZERO-SHOT ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ------------------------------------------------------------------------------
print("\n[5/7] Performing Zero-Shot Aspect Extraction...")
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=device
)

ASPECT_CANDIDATES = ["Food & Drinks", "Customer Service", "Value & Price", "Ambiance & Atmosphere"]

def extract_aspects(text, threshold=0.35):
    res = zero_shot_classifier(str(text), ASPECT_CANDIDATES, multi_label=True)
    detected = [label for label, score in zip(res['labels'], res['scores']) if score >= threshold]
    return detected if detected else ["General"]

tqdm.pandas()
df['detected_aspects'] = df[TEXT_COL].astype(str).progress_apply(extract_aspects)

# Explode dataset per aspect category for fine-grained reporting
df_absa = df.explode('detected_aspects').rename(columns={'detected_aspects': 'aspect'})