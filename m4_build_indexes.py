# Milestone 4: Build two FAISS indexes.
# Index 1: Similar tickets (body + answer pairs from dataset)
# Index 2: Policy documents (the 6 .md files in policy_kb/)
# Both saved to disk so we never rebuild them again.

import os
import json
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ---- Load embedding model (same one we used in M3) ----
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================
# INDEX 1: Similar Ticket Index
# ==============================================================
print("\n--- Building Similar Ticket Index ---")

# Load dataset
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
df = df[df["language"] == "en"].reset_index(drop=True)
df = df.dropna(subset=["body", "answer", "queue"]).reset_index(drop=True)

# Apply same category merge as before
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

# Deduplication -- remove near-identical ticket bodies
# We do this by truncating body to first 100 chars and dropping exact duplicates.
# Simple but effective for this synthetic dataset.
before = len(df)
df["body_short"] = df["body"].str[:100].str.strip().str.lower()
df = df.drop_duplicates(subset=["body_short"]).reset_index(drop=True)
df = df.drop(columns=["body_short"])
after = len(df)
print(f"Deduplication: {before} -> {after} tickets ({before - after} near-duplicates removed)")

# We'll use a sample of 10,000 tickets for the index to keep build time reasonable.
# In a real system you'd use all of them, but 10k is plenty for a demo.
df_sample = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
print(f"Building index on {len(df_sample)} tickets...")

# Embed ticket bodies (this takes a few minutes)
ticket_embeddings = embedder.encode(
    df_sample["body"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # normalizing makes cosine similarity work correctly
)

# Build FAISS index
# IndexFlatIP = exact search using inner product (= cosine similarity when normalized)
dimension = ticket_embeddings.shape[1]  # 384
ticket_index = faiss.IndexFlatIP(dimension)
ticket_index.add(ticket_embeddings.astype(np.float32))
print(f"Ticket index built: {ticket_index.ntotal} vectors, {dimension} dimensions")

# Save index and metadata
faiss.write_index(ticket_index, "ticket_index.faiss")

# Save the ticket bodies and answers so we can look them up by index position later
ticket_metadata = df_sample[["body", "answer", "queue_merged"]].to_dict(orient="records")
with open("ticket_metadata.json", "w") as f:
    json.dump(ticket_metadata, f)

print("Saved: ticket_index.faiss + ticket_metadata.json")

# ==============================================================
# INDEX 2: Policy KB Index
# ==============================================================
print("\n--- Building Policy KB Index ---")

POLICY_DIR = "policy_kb"
policy_chunks = []  # each chunk = one policy doc (small enough to not need splitting)

for filename in os.listdir(POLICY_DIR):
    if filename.endswith(".md"):
        filepath = os.path.join(POLICY_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        policy_chunks.append({
            "filename": filename,
            "content": content
        })
        print(f"  Loaded: {filename} ({len(content)} chars)")

print(f"\nTotal policy documents: {len(policy_chunks)}")

# Embed policy documents
policy_texts = [chunk["content"] for chunk in policy_chunks]
policy_embeddings = embedder.encode(
    policy_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Build FAISS index (tiny -- only 6 documents)
policy_index = faiss.IndexFlatIP(dimension)
policy_index.add(policy_embeddings.astype(np.float32))
print(f"Policy index built: {policy_index.ntotal} vectors")

# Save
faiss.write_index(policy_index, "policy_index.faiss")
with open("policy_metadata.json", "w") as f:
    json.dump(policy_chunks, f)

print("Saved: policy_index.faiss + policy_metadata.json")

# ==============================================================
# SANITY CHECK: Query both indexes with a sample ticket
# ==============================================================
print("\n--- Sanity Check: Query both indexes ---")

query = "I was charged twice for my subscription this month. Please help."
print(f"Query: '{query}'")

query_embedding = embedder.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

# Query similar ticket index -- top 3
print("\n=== Top 3 similar past tickets ===")
scores, indices = ticket_index.search(query_embedding, k=3)
for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    ticket = ticket_metadata[idx]
    print(f"\nRank {rank+1} (similarity: {score:.3f})")
    print(f"  Queue: {ticket['queue_merged']}")
    print(f"  Body:  {ticket['body'][:150]}...")
    print(f"  Answer: {ticket['answer'][:150]}...")

# Query policy index -- top 2
print("\n=== Top 2 relevant policy documents ===")
scores, indices = policy_index.search(query_embedding, k=2)
for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    chunk = policy_chunks[idx]
    print(f"\nRank {rank+1} (similarity: {score:.3f})")
    print(f"  File: {chunk['filename']}")
    print(f"  Content: {chunk['content'][:200]}...")

print("\nDone. Both indexes are ready for Milestone 5.")