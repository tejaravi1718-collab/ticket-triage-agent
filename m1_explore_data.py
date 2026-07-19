# Milestone 1: Just look at the data. No modeling yet.

import pandas as pd

# This loads the dataset directly from Hugging Face.
# (requires: pip install datasets --break-system-packages)
from datasets import load_dataset

ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()

# Keep English only, as we decided
df = df[df["language"] == "en"].reset_index(drop=True)

print(f"Total English tickets: {len(df)}")
print()
print("Columns available:", list(df.columns))
print()

# Look at the label distributions
print("=== Queue (category) distribution ===")
print(df["queue"].value_counts())
print()

print("=== Priority distribution ===")
print(df["priority"].value_counts())
print()

# Print 10 random full examples so you can READ them yourself
print("=== 10 random tickets ===")
sample = df.sample(10, random_state=42)
for i, row in sample.iterrows():
    print("-" * 60)
    print(f"SUBJECT: {row['subject']}")
    print(f"BODY: {row['body'][:300]}...")  # truncated for readability
    print(f"QUEUE: {row['queue']}  |  PRIORITY: {row['priority']}")