# ------------------------------------------------------------------------------
# SECTION 6: ROOT-CAUSE TOPIC MODELING (TF-IDF + NMF)
# ------------------------------------------------------------------------------
print("\n[6/7] Running Root-Cause Topic Discovery on negative feedback...")
neg_mask = df['roberta_label'].str.lower().str.contains('negative')
negative_reviews = df[neg_mask][TEXT_COL].astype(str).tolist()

if len(negative_reviews) > 0:
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(negative_reviews)
    
    num_topics = min(3, len(negative_reviews))
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    
    print("--> Discovered Problem Clusters:")
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
        print(f"    * Cluster #{topic_idx + 1}: {', '.join(top_words)}")
else:
    print("--> No negative reviews detected for topic clustering.")