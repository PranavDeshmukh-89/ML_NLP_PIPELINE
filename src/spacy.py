# ------------------------------------------------------------------------------
# SECTION 4: SPACY NOUN & MENU ITEM EXTRACTION
# ------------------------------------------------------------------------------
print("\n[4/7] Extracting menu items and entities with spaCy...")
nlp = spacy.load("en_core_web_sm")

def extract_aspect_entities(text):
    doc = nlp(str(text))
    extracted_items = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in ['NOUN', 'PROPN']:
            clean_chunk = " ".join([token.text for token in chunk if not token.is_stop])
            if len(clean_chunk.strip()) > 2:
                extracted_items.append(clean_chunk.lower())
    return extracted_items

df['extracted_entities'] = df[TEXT_COL].astype(str).apply(extract_aspect_entities)