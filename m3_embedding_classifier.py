# Milestone 3: Swap TF-IDF for sentence embeddings, compare against baseline.
# Goal: test whether better features fix the IT Support vs Technical Support confusion.

import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
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

# ---- Step 2: Same stratified split as Milestone 2 ----
# Using same random_state=42 so train/test sets are identical -- important for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    df["body"], df["queue"],
    test_size=0.2,
    random_state=42,
    stratify=df["queue"]
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- Step 3: Embed with all-MiniLM-L6-v2 ----
# This will take 5-15 minutes on CPU. The progress bar shows it's working -- don't panic.
print("\nLoading embedding model (downloads once, cached after)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("\nEmbedding training tickets (this takes a few minutes on CPU)...")
X_train_emb = embedder.encode(
    X_train.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding test tickets...")
X_test_emb = embedder.encode(
    X_test.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"\nEmbedding shape: {X_train_emb.shape}")
# You should see (22608, 384) -- 22608 tickets, each as 384 numbers

# ---- Step 4: Same classifier as before, new features ----
print("\nTraining classifier on embeddings...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_emb, y_train)

# ---- Step 5: Evaluate ----
y_pred = clf.predict(X_test_emb)

print()
print("=== Classification Report (Embedding-based) ===")
report = classification_report(y_test, y_pred)
print(report)

# ---- Step 6: Confusion matrix ----
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Queue Classification (Embeddings)")
plt.tight_layout()
plt.savefig("m3_confusion_matrix.png")
print("Confusion matrix saved as m3_confusion_matrix.png")

# ---- Step 7: Side-by-side comparison summary ----
# Paste your M2 numbers below to compare at a glance.
print()
print("=== Quick comparison ===")
print("Baseline (TF-IDF)  accuracy: 0.48   macro-F1: 0.47")
print(f"Embeddings         accuracy: see above -- check macro avg F1 in the report")
print()
print("Key question to answer:")
print("1. Did IT Support vs Technical Support confusion improve?")
print("2. Did the rare class over-prediction (General Inquiry) get better or worse?")
print("3. Which categories improved the most in F1?")