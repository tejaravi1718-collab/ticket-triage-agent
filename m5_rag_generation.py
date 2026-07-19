# Milestone 5: Generate grounded answers using retrieved context + Groq LLM.
# This is the "G" in RAG -- Retrieval Augmented Generation.
# We test both generation modes: similar-ticket and policy-based.

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ---- Load API key from .env file ----
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please check your .env file.")

groq_client = Groq(api_key=api_key)

# ---- Load embedding model and indexes ----
print("Loading embedding model and indexes...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

ticket_index = faiss.read_index("ticket_index.faiss")
policy_index = faiss.read_index("policy_index.faiss")

with open("ticket_metadata.json", "r") as f:
    ticket_metadata = json.load(f)

with open("policy_metadata.json", "r") as f:
    policy_metadata = json.load(f)

print("All indexes loaded.")

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def embed_query(text):
    """Turn a text query into a normalized embedding vector."""
    return embedder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)


def retrieve_similar_tickets(query, k=3):
    """Find top-k most similar past tickets."""
    query_vec = embed_query(query)
    scores, indices = ticket_index.search(query_vec, k=k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "body": ticket_metadata[idx]["body"],
            "answer": ticket_metadata[idx]["answer"],
            "queue": ticket_metadata[idx]["queue_merged"]
        })
    return results


def retrieve_policy(query, k=1):
    """Find the most relevant policy document."""
    query_vec = embed_query(query)
    scores, indices = policy_index.search(query_vec, k=k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "filename": policy_metadata[idx]["filename"],
            "content": policy_metadata[idx]["content"]
        })
    return results


def generate_with_similar_tickets(ticket_body, similar_tickets):
    """
    Generation mode 1: use past resolved tickets as context.
    Good for routine complaints where we have strong historical precedent.
    """
    context_block = "\n\n".join([
        f"Past ticket {i+1}:\nCustomer: {t['body'][:300]}\nAgent reply: {t['answer'][:300]}"
        for i, t in enumerate(similar_tickets)
    ])

    prompt = f"""You are a professional customer support agent. 
Use ONLY the past resolved tickets below as reference to draft a reply to the new ticket.
Do not make up information not present in the context.
Keep the reply professional, concise, and helpful.

PAST RESOLVED TICKETS (for reference):
{context_block}

NEW CUSTOMER TICKET:
{ticket_body}

Draft a professional reply to the new ticket:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # low temperature = more consistent, less creative
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def generate_with_policy(ticket_body, policy_results):
    """
    Generation mode 2: use policy document as context.
    Good for specific policy questions where accuracy matters most.
    """
    policy_block = "\n\n".join([
        f"Policy document ({p['filename']}):\n{p['content']}"
        for p in policy_results
    ])

    prompt = f"""You are a professional customer support agent.
Use ONLY the policy document below to answer the customer's question.
Do not make up information not present in the policy.
If the policy doesn't cover the question, say so honestly.
Keep the reply professional, concise, and helpful.

POLICY DOCUMENT:
{policy_block}

CUSTOMER TICKET:
{ticket_body}

Draft a professional reply based on the policy:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# ==============================================================
# TEST BOTH MODES WITH REAL EXAMPLES
# ==============================================================

print("\n" + "="*60)
print("TEST 1: Similar-ticket mode (routine billing complaint)")
print("="*60)

ticket_1 = "I was charged twice for my subscription this month and need this fixed immediately."

similar = retrieve_similar_tickets(ticket_1, k=3)
print(f"\nRetrieved {len(similar)} similar tickets (top similarity: {similar[0]['score']:.3f})")

answer_1 = generate_with_similar_tickets(ticket_1, similar)
print(f"\nTicket: {ticket_1}")
print(f"\nGenerated reply:\n{answer_1}")

# ----------------------------------------------------------

print("\n" + "="*60)
print("TEST 2: Policy mode (specific policy question)")
print("="*60)

ticket_2 = "How long do I have to raise a billing dispute? And how long does it take to resolve?"

policy = retrieve_policy(ticket_2, k=1)
print(f"\nRetrieved policy: {policy[0]['filename']} (similarity: {policy[0]['score']:.3f})")

answer_2 = generate_with_policy(ticket_2, policy)
print(f"\nTicket: {ticket_2}")
print(f"\nGenerated reply:\n{answer_2}")

# ----------------------------------------------------------

print("\n" + "="*60)
print("TEST 3: Policy mode (security/escalation question)")
print("="*60)

ticket_3 = "We think our account has been compromised. Someone accessed our data without permission."

policy = retrieve_policy(ticket_3, k=1)
print(f"\nRetrieved policy: {policy[0]['filename']} (similarity: {policy[0]['score']:.3f})")

answer_3 = generate_with_policy(ticket_3, policy)
print(f"\nTicket: {ticket_3}")
print(f"\nGenerated reply:\n{answer_3}")

print("\n" + "="*60)
print("Milestone 5 complete.")
print("Both generation modes are working.")
print("Next: Milestone 6 -- the agent that decides WHICH mode to use.")
print("="*60)