# Save the trained classifier and vectorizer to disk.
# Run this once after m3b_merged_classifier.py -- saves the model so we never
# have to retrain from scratch again. Every future milestone just loads these files.

import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---- Same pipeline as m3b -- must be identical or loaded model won't match ----
print("Loading dataset...")
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
df = df[df["language"] == "en"].reset_index(drop=True)
df = df.dropna(subset=["body", "queue"]).reset_index(drop=True)

# Same merge map as m3b
CATEGORY_MERGE_MAP = {
    "IT Support":                     "Technical & IT Support",
    "Technical Support":              "Technical & IT Support",
    "Customer Service":               "Customer & Sales Support",
    "General Inquiry":                "Customer & Sales Support",
    "Sales and Pre-Sales":            "Customer & Sales Support",
    "Billing and Payments":           "Billing and Payments",
    "Human Resources":                "Human Resources",
    "Product Support":                "Product Support",
    "Returns and Exchanges":          "Returns and Exchanges",
    "Service Outages and Maintenance": "Service Outages and Maintenance",
}
df["queue_merged"] = df["queue"].map(CATEGORY_MERGE_MAP)

# Same split
X_train, X_test, y_train, y_test = train_test_split(
    df["body"], df["queue_merged"],
    test_size=0.2,
    random_state=42,
    stratify=df["queue_merged"]
)

# Same vectorizer and classifier
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)

# ---- Save both to disk ----
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(clf, "queue_classifier.pkl")

print()
print("Saved:")
print("  tfidf_vectorizer.pkl  -- the TF-IDF vectorizer (text -> numbers)")
print("  queue_classifier.pkl  -- the trained Logistic Regression classifier")
print()
print("To load and use in any future script:")
print("  import joblib")
print("  vectorizer = joblib.load('tfidf_vectorizer.pkl')")
print("  clf        = joblib.load('queue_classifier.pkl')")
print()

# ---- Quick sanity check: predict one ticket manually ----
sample = "My software keeps crashing after the latest update and I cannot open the application."
vec = vectorizer.transform([sample])
pred = clf.predict(vec)[0]
proba = clf.predict_proba(vec).max()

print("=== Sanity check ===")
print(f"Ticket: '{sample}'")
print(f"Predicted queue: {pred}")
print(f"Confidence: {proba:.2f}")
print()
print("Does that prediction make sense to you? It should be Technical & IT Support.")