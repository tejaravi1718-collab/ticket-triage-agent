# Milestone 7, Part 2: Run the agent on all 50 labeled tickets and compute metrics.
# Run this AFTER you've filled in eval_tickets_to_label.csv.

import os
import json
import joblib
import numpy as np
import pandas as pd
import faiss
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from groq import Groq
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Load all artifacts ----
print("Loading artifacts...")
vectorizer   = joblib.load("tfidf_vectorizer.pkl")
clf          = joblib.load("queue_classifier.pkl")
embedder     = SentenceTransformer("all-MiniLM-L6-v2")
ticket_index = faiss.read_index("ticket_index.faiss")
policy_index = faiss.read_index("policy_index.faiss")

with open("ticket_metadata.json") as f: ticket_metadata = json.load(f)
with open("policy_metadata.json") as f: policy_metadata = json.load(f)

# ---- Priority inference (same as m6) ----
CRITICAL_KEYWORDS = ["breach", "hacked", "unauthorized access", "data leak", "outage", "down for everyone"]
HIGH_KEYWORDS     = ["charged twice", "charged three times", "double charge", "cannot access", "urgent", "immediately"]

def infer_priority(text):
    t = text.lower()
    if any(k in t for k in CRITICAL_KEYWORDS): return "critical"
    if any(k in t for k in HIGH_KEYWORDS):     return "high"
    if "?" in text and len(text) < 200:        return "low"
    return "medium"

# ---- AgentState ----
class AgentState(TypedDict):
    ticket_body:             str
    queue:                   Optional[str]
    priority:                Optional[str]
    route:                   Optional[str]
    retrieved_context:       List[dict]
    generated_reply:         Optional[str]
    needs_clarification:     bool
    clarification_question:  Optional[str]
    log:                     dict

# ---- Node functions (identical to m6) ----
def classify_ticket(state):
    body  = state["ticket_body"]
    vec   = vectorizer.transform([body])
    queue = clf.predict(vec)[0]
    conf  = clf.predict_proba(vec).max()
    priority = infer_priority(body)
    return {**state, "queue": queue, "priority": priority,
            "log": {**state["log"], "queue": queue, "priority": priority, "classifier_conf": round(conf, 3)}}

def route_decision(state):
    priority = state["priority"]
    body     = state["ticket_body"]
    body_low = body.lower()
    if priority == "critical":
        route = "escalate"
    elif any(k in body_low for k in [
        "how long", "what is your policy", "refund policy", "sla", "response time",
        "within how many days", "what are the terms", "policy", "procedure"
    ]):
        route = "policy_kb"
    elif len(body.strip()) < 60 or (body.strip().endswith("?") and len(body) < 80):
        route = "ask_clarifying"
    else:
        route = "similar_tickets"
    return {**state, "route": route, "log": {**state["log"], "route": route}}

def retrieve_similar_tickets(state):
    body      = state["ticket_body"]
    query_vec = embedder.encode([body], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = ticket_index.search(query_vec, k=3)
    context = [{"type": "similar_ticket", "score": round(float(s), 3),
                 "body": ticket_metadata[i]["body"], "answer": ticket_metadata[i]["answer"],
                 "queue": ticket_metadata[i]["queue_merged"]}
                for s, i in zip(scores[0], indices[0])]
    return {**state, "retrieved_context": context,
            "log": {**state["log"], "retrieval_type": "similar_tickets", "top_score": context[0]["score"]}}

def retrieve_policy_kb(state):
    body      = state["ticket_body"]
    query_vec = embedder.encode([body], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = policy_index.search(query_vec, k=2)
    context = [{"type": "policy", "score": round(float(s), 3),
                 "filename": policy_metadata[i]["filename"], "content": policy_metadata[i]["content"]}
                for s, i in zip(scores[0], indices[0])]
    return {**state, "retrieved_context": context,
            "log": {**state["log"], "retrieval_type": "policy_kb",
                    "top_doc": context[0]["filename"], "top_score": context[0]["score"]}}

def escalate(state):
    reply = ("Your ticket has been flagged as critical priority and escalated to a senior "
             "support agent immediately. You will be contacted within 1 hour.")
    return {**state, "generated_reply": reply, "log": {**state["log"], "escalated": True}}

def ask_clarifying_question(state):
    question = ("To help you as quickly as possible, could you please provide more details? "
                "Which product or service is affected, what error you're seeing, and when it started.")
    return {**state, "needs_clarification": True, "clarification_question": question,
            "generated_reply": question, "log": {**state["log"], "needs_clarification": True}}

def generate_reply(state):
    body    = state["ticket_body"]
    context = state["retrieved_context"]
    if not context:
        return {**state, "generated_reply": "We have received your ticket and will respond shortly."}
    if context[0]["type"] == "similar_ticket":
        context_block = "\n\n".join([
            f"Past ticket {i+1}:\nCustomer: {c['body'][:300]}\nAgent: {c['answer'][:300]}"
            for i, c in enumerate(context)])
        prompt = f"You are a support agent. Use ONLY these past tickets as reference.\n\n{context_block}\n\nNew ticket: {body}\n\nDraft a professional reply:"
    else:
        context_block = "\n\n".join([f"Policy ({c['filename']}):\n{c['content']}" for c in context])
        prompt = f"You are a support agent. Use ONLY this policy to answer.\n\n{context_block}\n\nTicket: {body}\n\nDraft a professional reply:"
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=300)
    return {**state, "generated_reply": response.choices[0].message.content.strip(),
            "log": {**state["log"], "reply_generated": True}}

def log_decision(state):
    return {**state, "log": {**state["log"], "ticket_body": state["ticket_body"][:100]}}

def get_next_node(state):
    route = state["route"]
    if route == "escalate":       return "escalate"
    if route == "ask_clarifying": return "ask_clarifying"
    if route == "policy_kb":      return "retrieve_policy_kb"
    return "retrieve_similar_tickets"

# ---- Build graph ----
graph = StateGraph(AgentState)
graph.add_node("classify_ticket",          classify_ticket)
graph.add_node("route_decision",           route_decision)
graph.add_node("retrieve_similar_tickets", retrieve_similar_tickets)
graph.add_node("retrieve_policy_kb",       retrieve_policy_kb)
graph.add_node("escalate",                 escalate)
graph.add_node("ask_clarifying",           ask_clarifying_question)
graph.add_node("generate_reply",           generate_reply)
graph.add_node("log_decision",             log_decision)
graph.set_entry_point("classify_ticket")
graph.add_edge("classify_ticket",          "route_decision")
graph.add_edge("retrieve_similar_tickets", "generate_reply")
graph.add_edge("retrieve_policy_kb",       "generate_reply")
graph.add_edge("generate_reply",           "log_decision")
graph.add_edge("escalate",                 "log_decision")
graph.add_edge("ask_clarifying",           "log_decision")
graph.add_edge("log_decision",             END)
graph.add_conditional_edges("route_decision", get_next_node, {
    "retrieve_similar_tickets": "retrieve_similar_tickets",
    "retrieve_policy_kb":       "retrieve_policy_kb",
    "escalate":                 "escalate",
    "ask_clarifying":           "ask_clarifying"
})
agent = graph.compile()
print("Agent compiled.\n")

# ---- Load labeled CSV ----
label_df = pd.read_csv("eval_tickets_to_label.csv")

# Check all rows are labeled
unlabeled = label_df[label_df["correct_route"].isna() | (label_df["correct_route"] == "")]
if len(unlabeled) > 0:
    print(f"WARNING: {len(unlabeled)} rows are not yet labeled (rows: {list(unlabeled['id'])}).")
    print("Please fill in correct_route and correct_queue for all rows before running eval.")
    exit()

print(f"Running agent on {len(label_df)} labeled tickets...")
print("(This will make LLM calls for retrieval tickets -- may take a few minutes)\n")

# ---- Run agent on every ticket ----
results = []
for _, row in label_df.iterrows():
    initial_state: AgentState = {
        "ticket_body": row["body"], "queue": None, "priority": None,
        "route": None, "retrieved_context": [], "generated_reply": None,
        "needs_clarification": False, "clarification_question": None, "log": {}
    }
    result = agent.invoke(initial_state)
    results.append({
        "id":               row["id"],
        "body":             row["body"][:100],
        "correct_route":    row["correct_route"].strip(),
        "correct_queue":    row["correct_queue"].strip(),
        "predicted_route":  result["route"],
        "predicted_queue":  result["queue"],
        "route_correct":    row["correct_route"].strip() == result["route"],
        "queue_correct":    row["correct_queue"].strip() == result["queue"],
        "classifier_conf":  result["log"].get("classifier_conf", None),
        "top_score":        result["log"].get("top_score", None),
    })
    print(f"  [{row['id']:02d}] route: {result['route']:22s} | {'✓' if row['correct_route'].strip() == result['route'] else '✗'} "
          f"| queue: {result['queue']}")

results_df = pd.DataFrame(results)
results_df.to_csv("eval_results.csv", index=False)

# ---- Compute metrics ----
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

routing_accuracy = results_df["route_correct"].mean()
queue_accuracy   = results_df["queue_correct"].mean()

print(f"\nRouting accuracy:   {routing_accuracy:.1%}  ({results_df['route_correct'].sum()}/{len(results_df)} correct)")
print(f"Queue accuracy:     {queue_accuracy:.1%}  ({results_df['queue_correct'].sum()}/{len(results_df)} correct)")

print("\n--- Routing breakdown by correct route ---")
for route in results_df["correct_route"].unique():
    subset = results_df[results_df["correct_route"] == route]
    acc = subset["route_correct"].mean()
    print(f"  {route:25s}: {acc:.1%}  ({subset['route_correct'].sum()}/{len(subset)})")

print("\n--- Misrouted tickets ---")
misrouted = results_df[~results_df["route_correct"]]
if len(misrouted) == 0:
    print("  None! Perfect routing accuracy.")
else:
    for _, row in misrouted.iterrows():
        print(f"  ID {row['id']:02d}: correct={row['correct_route']} | predicted={row['predicted_route']}")
        print(f"         body: {row['body'][:80]}...")

print("\n--- Queue classification report ---")
print(classification_report(results_df["correct_queue"], results_df["predicted_queue"], zero_division=0))

# ---- Routing confusion matrix ----
routes = sorted(results_df["correct_route"].unique())
cm = confusion_matrix(results_df["correct_route"], results_df["predicted_route"], labels=routes)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=routes, yticklabels=routes, cmap="Blues")
plt.xlabel("Predicted route")
plt.ylabel("Correct route")
plt.title("Routing Confusion Matrix (Eval Set)")
plt.tight_layout()
plt.savefig("m7_routing_confusion.png")
print("\nRouting confusion matrix saved as m7_routing_confusion.png")
print("\nFull results saved to eval_results.csv")
print("\n" + "=" * 60)
print("RESUME BULLET TEMPLATE")
print("=" * 60)
print(f"Evaluated agent routing decisions against a 50-ticket hand-labeled")
print(f"test set, achieving {routing_accuracy:.0%} routing accuracy across 4 decision")
print(f"branches (similar_tickets / policy_kb / escalate / ask_clarifying).")