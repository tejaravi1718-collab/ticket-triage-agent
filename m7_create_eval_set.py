# Milestone 7, Part 1: Sample 50 tickets and create a CSV for hand-labeling.
# The first 4 rows are the M6 test tickets (you already know their correct routes).
# Fill in the empty 'correct_route' and 'correct_queue' columns, then run m7_eval.py.

import pandas as pd
import json
from datasets import load_dataset

# ---- M6 test tickets (pre-filled -- you already labeled these) ----
m6_tickets = [
    {
        "id": 1,
        "body": "I was charged twice for my subscription this month and need this fixed immediately.",
        "correct_route": "similar_tickets",
        "correct_queue": "Billing and Payments",
        "retrieved_relevant": ""  # fill after running eval
    },
    {
        "id": 2,
        "body": "What is your refund policy? How long do I have to request one?",
        "correct_route": "policy_kb",
        "correct_queue": "Customer & Sales Support",
        "retrieved_relevant": ""
    },
    {
        "id": 3,
        "body": "We think our account was hacked. Someone accessed our data without authorization.",
        "correct_route": "escalate",
        "correct_queue": "Technical & IT Support",
        "retrieved_relevant": "N/A"  # escalate doesn't retrieve
    },
    {
        "id": 4,
        "body": "Hi, it's not working.",
        "correct_route": "ask_clarifying",
        "correct_queue": "Technical & IT Support",
        "retrieved_relevant": "N/A"  # clarifying doesn't retrieve
    },
]

# ---- Sample 46 more tickets from the dataset ----
print("Loading dataset...")
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
df = df[df["language"] == "en"].reset_index(drop=True)
df = df.dropna(subset=["body", "queue"]).reset_index(drop=True)

# Apply same category merge
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

# Sample 46 tickets -- stratified so we get variety across queues
# ~6-7 per category (7 categories x 6 = 42, round up to 46)
sampled = []
for queue in df["queue_merged"].unique():
    subset = df[df["queue_merged"] == queue].sample(
        n=min(7, len(df[df["queue_merged"] == queue])),
        random_state=42
    )
    sampled.append(subset)

sampled_df = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
sampled_df = sampled_df.head(46)

# Build the rows for the CSV
dataset_rows = []
for i, row in sampled_df.iterrows():
    dataset_rows.append({
        "id": i + 5,  # starts at 5 (after the 4 M6 tickets)
        "body": row["body"][:400],  # truncated for readability
        "correct_route": "",  # YOU fill this in
        "correct_queue": "",  # YOU fill this in
        "retrieved_relevant": ""  # fill after running eval
    })

# ---- Combine M6 tickets + sampled tickets ----
all_rows = m6_tickets + dataset_rows
label_df = pd.DataFrame(all_rows)
label_df.to_csv("eval_tickets_to_label.csv", index=False)

print(f"Created eval_tickets_to_label.csv with {len(label_df)} tickets.")
print()
print("=" * 60)
print("INSTRUCTIONS FOR LABELING")
print("=" * 60)
print()
print("Open eval_tickets_to_label.csv and fill in two columns for")
print("rows 5 onwards (rows 1-4 are already filled from M6):")
print()
print("1. correct_route -- choose ONE of:")
print("   similar_tickets   (routine complaint, has enough detail)")
print("   policy_kb         (asks about policy, rules, timelines)")
print("   escalate          (security incident, data breach, critical)")
print("   ask_clarifying    (too vague, missing key details)")
print()
print("2. correct_queue -- choose ONE of:")
print("   Billing and Payments")
print("   Customer & Sales Support")
print("   Human Resources")
print("   Product Support")
print("   Returns and Exchanges")
print("   Service Outages and Maintenance")
print("   Technical & IT Support")
print()
print("TIPS FOR LABELING:")
print("- Read the full body text, not just the subject")
print("- If a ticket mentions 'breach', 'hacked', 'unauthorized' -> escalate")
print("- If a ticket asks 'how long', 'what is the policy' -> policy_kb")
print("- If the ticket body is vague (under ~2 sentences, no specifics) -> ask_clarifying")
print("- Everything else -> similar_tickets")
print()
print("Once done, save the CSV and run: python m7_eval.py")