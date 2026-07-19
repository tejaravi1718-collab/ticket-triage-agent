# Milestone 2: Simplest possible classifier for ticket "queue" (category).
# No tuning, no fancy stuff -- just a working baseline.

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Step 1: Load and filter data (same as Milestone 1) ----
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
df = df[df["language"] == "en"].reset_index(drop=True)

# Drop rows with missing body or queue -- can't learn from empty text/labels
df = df.dropna(subset=["body", "queue"]).reset_index(drop=True)
print(f"Tickets after dropping missing body/queue: {len(df)}")

# ---- Step 2: Split into train and test sets ----
# stratify=df["queue"] makes sure each split has the same proportion
# of each category as the full dataset -- important given the imbalance we found.
X_train, X_test, y_train, y_test = train_test_split(
    df["body"], df["queue"],
    test_size=0.2,
    random_state=42,
    stratify=df["queue"]
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Step 3: Turn text into numbers (TF-IDF) ----
# max_features limits vocabulary size to the 5000 most informative words/phrases
# ngram_range=(1,2) means it looks at single words AND two-word phrases
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- Step 4: Train the simplest classifier ----
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
# class_weight="balanced" tells the model to pay MORE attention to rare classes
# (like General Inquiry) instead of just favoring the common ones (Technical Support).
clf.fit(X_train_vec, y_train)

# ---- Step 5: Evaluate ----
y_pred = clf.predict(X_test_vec)

print()
print("=== Classification Report (per-class precision/recall/F1) ===")
print(classification_report(y_test, y_pred))

# ---- Step 6: Confusion matrix, saved as an image to actually look at ----
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Queue Classification (Baseline)")
plt.tight_layout()
plt.savefig("m2_confusion_matrix.png")
print()
print("Confusion matrix saved as m2_confusion_matrix.png -- open it and look.")