# Milestone 3b: Merge overlapping categories, retrain with TF-IDF.
# Based on confusion matrix evidence from M2 and M3.

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Step 1: Load and filter (same as before) ----
print("Loading dataset...")
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
df = df[df["language"] == "en"].reset_index(drop=True)
df = df.dropna(subset=["body", "queue"]).reset_index(drop=True)
print(f"Tickets loaded: {len(df)}")

# ---- Step 2: Merge overlapping categories ----
# This is just a data relabeling step -- no model change yet.

CATEGORY_MERGE_MAP = {
    # Merge 1: IT Support + Technical Support (bled into each other heavily)
    "IT Support":         "Technical & IT Support",
    "Technical Support":  "Technical & IT Support",

    # Merge 2: Customer Service + General Inquiry + Sales and Pre-Sales
    # (all low-precision, semantically overlapping customer-facing queries)
    "Customer Service":   "Customer & Sales Support",
    "General Inquiry":    "Customer & Sales Support",
    "Sales and Pre-Sales": "Customer & Sales Support",

    # Everything else stays as-is
    "Billing and Payments":           "Billing and Payments",
    "Human Resources":                "Human Resources",
    "Product Support":                "Product Support",
    "Returns and Exchanges":          "Returns and Exchanges",
    "Service Outages and Maintenance": "Service Outages and Maintenance",
}

df["queue_merged"] = df["queue"].map(CATEGORY_MERGE_MAP)

# Verify the merge -- print before/after counts
print()
print("=== Before merge (original 10 categories) ===")
print(df["queue"].value_counts())
print()
print("=== After merge (7 categories) ===")
print(df["queue_merged"].value_counts())

# ---- Step 3: Same stratified split, now on merged labels ----
X_train, X_test, y_train, y_test = train_test_split(
    df["body"], df["queue_merged"],
    test_size=0.2,
    random_state=42,
    stratify=df["queue_merged"]
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ---- Step 4: TF-IDF (back to what worked) ----
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- Step 5: Same classifier ----
print("\nTraining classifier...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)

# ---- Step 6: Evaluate ----
y_pred = clf.predict(X_test_vec)

print()
print("=== Classification Report (Merged categories, TF-IDF) ===")
print(classification_report(y_test, y_pred))

# ---- Step 7: Confusion matrix ----
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Merged Categories (TF-IDF)")
plt.tight_layout()
plt.savefig("m3b_confusion_matrix.png")
print("Confusion matrix saved as m3b_confusion_matrix.png")

# ---- Step 8: Direct comparison summary ----
print()
print("=== Full comparison across all three runs ===")
print("Run                              Accuracy   Macro-F1")
print("M2: TF-IDF, 10 categories          0.48       0.47")
print("M3: Embeddings, 10 categories      0.31       0.30")
print("M3b: TF-IDF, 7 merged categories   ???        ???  <- fill in from report above")
print()
print("Key questions:")
print("1. Did overall accuracy and macro-F1 improve over the M2 baseline?")
print("2. Is the new 'Technical & IT Support' category now much cleaner?")
print("3. Is 'Customer & Sales Support' cleaner than the three categories it replaced?")
print("4. Did any of the unchanged categories (Billing, HR, etc.) get worse?")