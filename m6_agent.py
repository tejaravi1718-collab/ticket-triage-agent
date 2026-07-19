# Milestone 6: LangGraph agent that orchestrates the full ticket triage pipeline.
# Nodes: classify -> route -> {similar_tickets | policy_kb | escalate | ask_clarifying} -> generate -> log

import os
import json
import joblib
import numpy as np
import faiss
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Load all saved artifacts ----
print("Loading models and indexes...")
vectorizer    = joblib.load("tfidf_vectorizer.pkl")
clf           = joblib.load("queue_classifier.pkl")
embedder      = SentenceTransformer("all-MiniLM-L6-v2")
ticket_index  = faiss.read_index("ticket_index.faiss")
policy_index  = faiss.read_index("policy_index.faiss")

with open("ticket_metadata.json")  as f: ticket_metadata = json.load(f)
with open("policy_metadata.json")  as f: policy_metadata = json.load(f)
print("All artifacts loaded.\n")

# ---- Priority map: the dataset uses text labels ----
# We'll simulate priority from classifier confidence + keywords for now.
# In a real system you'd train a separate priority classifier.
CRITICAL_KEYWORDS = ["breach", "hacked", "unauthorized access", "data leak", "outage", "down for everyone"]
HIGH_KEYWORDS     = ["charged twice", "charged three times", "double charge", "cannot access", "urgent", "immediately"]

def infer_priority(text: str) -> str:
    t = text.lower()
    if any(k in t for k in CRITICAL_KEYWORDS): return "critical"
    if any(k in t for k in HIGH_KEYWORDS):     return "high"
    if "?" in text and len(text) < 200:        return "low"
    return "medium"

# ---- AgentState: the shared notepad every node reads and writes ----
class AgentState(TypedDict):
    ticket_body:          str
    queue:                Optional[str]
    priority:             Optional[str]
    route:                Optional[str]
    retrieved_context:    List[dict]
    generated_reply:      Optional[str]
    needs_clarification:  bool
    clarification_question: Optional[str]
    log:                  dict

# ==============================================================
# NODE FUNCTIONS
# Each node receives the full state and returns updated fields.
# ==============================================================

def classify_ticket(state: AgentState) -> AgentState:
    """Node 1: Predict queue and priority from ticket text."""
    body = state["ticket_body"]

    # Queue prediction (ML model)
    vec  = vectorizer.transform([body])
    queue = clf.predict(vec)[0]
    conf  = clf.predict_proba(vec).max()

    # Priority (keyword heuristic -- replace with trained model if desired)
    priority = infer_priority(body)

    print(f"[CLASSIFY] Queue: {queue} (conf: {conf:.2f}) | Priority: {priority}")

    return {
        **state,
        "queue":    queue,
        "priority": priority,
        "log": {**state["log"], "queue": queue, "priority": priority, "classifier_conf": round(conf, 3)}
    }


def route_decision(state: AgentState) -> AgentState:
    priority = state["priority"]
    body     = state["ticket_body"]
    body_low = body.lower()

    # Rule 1: Critical -> escalate immediately (unchanged)
    if priority == "critical":
        route = "escalate"

    # Rule 2: Policy keywords -> policy KB (MOVED UP before clarifying check)
    elif any(k in body_low for k in [
        "how long", "what is your policy", "refund policy", "sla", "response time",
        "within how many days", "what are the terms", "policy", "procedure"
    ]):
        route = "policy_kb"

    # Rule 3: Too vague -> ask clarifying (MOVED DOWN)
    elif len(body.strip()) < 60 or body.strip().endswith("?") and len(body) < 80:
        route = "ask_clarifying"

    # Rule 4: Everything else -> similar tickets (unchanged)
    else:
        route = "similar_tickets"

    print(f"[ROUTE]    Decision: {route}")
    return {**state, "route": route, "log": {**state["log"], "route": route}}

def retrieve_similar_tickets(state: AgentState) -> AgentState:
    """Node 3a: Find top-3 similar past resolved tickets."""
    body = state["ticket_body"]
    query_vec = embedder.encode([body], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = ticket_index.search(query_vec, k=3)

    context = []
    for score, idx in zip(scores[0], indices[0]):
        context.append({
            "type":   "similar_ticket",
            "score":  round(float(score), 3),
            "body":   ticket_metadata[idx]["body"],
            "answer": ticket_metadata[idx]["answer"],
            "queue":  ticket_metadata[idx]["queue_merged"]
        })

    print(f"[RETRIEVE] Similar tickets: top similarity {context[0]['score']:.3f}")
    return {**state, "retrieved_context": context, "log": {**state["log"], "retrieval_type": "similar_tickets", "top_score": context[0]["score"]}}


def retrieve_policy_kb(state: AgentState) -> AgentState:
    """Node 3b: Find the most relevant policy document."""
    body = state["ticket_body"]
    query_vec = embedder.encode([body], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = policy_index.search(query_vec, k=2)

    context = []
    for score, idx in zip(scores[0], indices[0]):
        context.append({
            "type":     "policy",
            "score":    round(float(score), 3),
            "filename": policy_metadata[idx]["filename"],
            "content":  policy_metadata[idx]["content"]
        })

    print(f"[RETRIEVE] Policy KB: top doc '{context[0]['filename']}' (score: {context[0]['score']:.3f})")
    return {**state, "retrieved_context": context, "log": {**state["log"], "retrieval_type": "policy_kb", "top_doc": context[0]["filename"], "top_score": context[0]["score"]}}


def escalate(state: AgentState) -> AgentState:
    """Node 3c: Flag ticket for human escalation -- no generation."""
    reply = (
        "Thank you for contacting us. Your ticket has been flagged as critical priority "
        "and escalated to a senior support agent immediately. You will be contacted within "
        "1 hour via a secure channel. Please do not share sensitive credentials in this ticket."
    )
    print(f"[ESCALATE] Ticket flagged for human agent.")
    return {
        **state,
        "generated_reply": reply,
        "log": {**state["log"], "escalated": True, "reply_generated": False}
    }


def ask_clarifying_question(state: AgentState) -> AgentState:
    """Node 3d: Request more information from the customer."""
    question = (
        "Thank you for reaching out. To help you as quickly as possible, could you please "
        "provide more details about the issue you're experiencing? For example: which product "
        "or service is affected, what error message (if any) you're seeing, and when the issue started."
    )
    print(f"[CLARIFY]  Requesting more information from customer.")
    return {
        **state,
        "needs_clarification":   True,
        "clarification_question": question,
        "generated_reply":       question,
        "log": {**state["log"], "needs_clarification": True, "reply_generated": False}
    }


def generate_reply(state: AgentState) -> AgentState:
    """Node 4: Generate a grounded reply using retrieved context."""
    body    = state["ticket_body"]
    context = state["retrieved_context"]

    if not context:
        return {**state, "generated_reply": "We have received your ticket and will respond shortly."}

    # Build prompt based on retrieval type
    if context[0]["type"] == "similar_ticket":
        context_block = "\n\n".join([
            f"Past resolved ticket {i+1}:\nCustomer: {c['body'][:300]}\nAgent reply: {c['answer'][:300]}"
            for i, c in enumerate(context)
        ])
        prompt = f"""You are a professional customer support agent.
Use ONLY the past resolved tickets below as reference to draft a reply.
Do not make up information not present in the context.
Keep the reply professional, concise, and helpful.

PAST RESOLVED TICKETS:
{context_block}

NEW CUSTOMER TICKET:
{body}

Draft a professional reply:"""

    else:  # policy
        context_block = "\n\n".join([
            f"Policy ({c['filename']}):\n{c['content']}"
            for c in context
        ])
        prompt = f"""You are a professional customer support agent.
Use ONLY the policy document below to answer the customer's question.
Do not make up information not in the policy.
Keep the reply professional, concise, and helpful.

POLICY:
{context_block}

CUSTOMER TICKET:
{body}

Draft a professional reply based on the policy:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    reply = response.choices[0].message.content.strip()
    print(f"[GENERATE] Reply generated ({len(reply)} chars)")
    return {**state, "generated_reply": reply, "log": {**state["log"], "reply_generated": True}}


def log_decision(state: AgentState) -> AgentState:
    """Node 5: Finalize the log entry for eval."""
    log = {**state["log"], "ticket_body": state["ticket_body"][:100]}
    print(f"[LOG]      Decision logged.")
    return {**state, "log": log}


# ---- Conditional edge: what does route_decision route to? ----
def get_next_node(state: AgentState) -> str:
    route = state["route"]
    if route == "escalate":        return "escalate"
    if route == "ask_clarifying":  return "ask_clarifying"
    if route == "policy_kb":       return "retrieve_policy_kb"
    return "retrieve_similar_tickets"


# ==============================================================
# BUILD THE GRAPH
# ==============================================================
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("classify_ticket",          classify_ticket)
graph.add_node("route_decision",           route_decision)
graph.add_node("retrieve_similar_tickets", retrieve_similar_tickets)
graph.add_node("retrieve_policy_kb",       retrieve_policy_kb)
graph.add_node("escalate",                 escalate)
graph.add_node("ask_clarifying",           ask_clarifying_question)
graph.add_node("generate_reply",           generate_reply)
graph.add_node("log_decision",             log_decision)

# Set entry point
graph.set_entry_point("classify_ticket")

# Fixed edges (always go to next node)
graph.add_edge("classify_ticket",          "route_decision")
graph.add_edge("retrieve_similar_tickets", "generate_reply")
graph.add_edge("retrieve_policy_kb",       "generate_reply")
graph.add_edge("generate_reply",           "log_decision")
graph.add_edge("escalate",                 "log_decision")
graph.add_edge("ask_clarifying",           "log_decision")
graph.add_edge("log_decision",             END)

# Conditional edge: route_decision -> one of four branches
graph.add_conditional_edges("route_decision", get_next_node, {
    "retrieve_similar_tickets": "retrieve_similar_tickets",
    "retrieve_policy_kb":       "retrieve_policy_kb",
    "escalate":                 "escalate",
    "ask_clarifying":           "ask_clarifying"
})

# Compile the graph
agent = graph.compile()
print("Agent graph compiled successfully.\n")

# ==============================================================
# RUN FOUR TEST CASES -- ONE PER BRANCH
# ==============================================================

test_tickets = [
    {
        "label": "TEST 1 — Routine complaint (similar_tickets branch)",
        "body":  "I was charged twice for my subscription this month and need this fixed immediately."
    },
    {
        "label": "TEST 2 — Policy question (policy_kb branch)",
        "body":  "What is your refund policy? How long do I have to request one?"
    },
    {
        "label": "TEST 3 — Security incident (escalate branch)",
        "body":  "We think our account was hacked. Someone accessed our data without authorization."
    },
    {
        "label": "TEST 4 — Vague ticket (ask_clarifying branch)",
        "body":  "Hi, it's not working."
    }
]

for test in test_tickets:
    print("=" * 60)
    print(test["label"])
    print("=" * 60)

    initial_state: AgentState = {
        "ticket_body":            test["body"],
        "queue":                  None,
        "priority":               None,
        "route":                  None,
        "retrieved_context":      [],
        "generated_reply":        None,
        "needs_clarification":    False,
        "clarification_question": None,
        "log":                    {}
    }

    result = agent.invoke(initial_state)

    print(f"\nTicket:  {test['body']}")
    print(f"Queue:   {result['queue']} | Priority: {result['priority']}")
    print(f"Route:   {result['route']}")
    print(f"Reply:\n{result['generated_reply']}")
    print()