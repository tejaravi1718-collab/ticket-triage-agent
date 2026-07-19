# Milestone 8: Streamlit UI for the Ticket Triage Agent
# Run locally: streamlit run app.py
# Deploy: push to Hugging Face Spaces

import os
import json
import joblib
import numpy as np
import faiss
import streamlit as st
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from groq import Groq

# ---- Page config ----
st.set_page_config(
    page_title="Ticket Triage Agent",
    page_icon="🎫",
    layout="centered"
)

# ---- Styling ----
st.markdown("""
<style>
    /* Main background and font */
    .main { background-color: #0f1117; }

    /* Step cards */
    .step-card {
        background: #1a1d27;
        border: 1px solid #2e3145;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .step-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }
    .step-value {
        font-size: 1.05rem;
        color: #e5e7eb;
        font-weight: 500;
    }

    /* Route badge colors */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-similar   { background: #1e3a5f; color: #60a5fa; }
    .badge-policy    { background: #1e3a2f; color: #34d399; }
    .badge-escalate  { background: #3a1e1e; color: #f87171; }
    .badge-clarify   { background: #3a2e1e; color: #fbbf24; }
    .badge-priority-critical { background: #3a1e1e; color: #f87171; }
    .badge-priority-high     { background: #3a2e1e; color: #fbbf24; }
    .badge-priority-medium   { background: #1e2e3a; color: #60a5fa; }
    .badge-priority-low      { background: #1e3a2f; color: #34d399; }

    /* Retrieved context box */
    .context-box {
        background: #13151f;
        border-left: 3px solid #3b4fd8;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        color: #9ca3af;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }

    /* Reply box */
    .reply-box {
        background: #13151f;
        border-left: 3px solid #34d399;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.92rem;
        color: #e5e7eb;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }

    /* Divider */
    hr { border-color: #2e3145; margin: 1.5rem 0; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Load env (local only -- HF Spaces uses Secrets) ----
load_dotenv()

# ==============================================================
# LOAD ALL ARTIFACTS (cached so they only load once)
# ==============================================================

@st.cache_resource
def load_artifacts():
    groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
    vectorizer   = joblib.load("tfidf_vectorizer.pkl")
    clf          = joblib.load("queue_classifier.pkl")
    embedder     = SentenceTransformer("all-MiniLM-L6-v2")
    t_index      = faiss.read_index("ticket_index.faiss")
    p_index      = faiss.read_index("policy_index.faiss")
    with open("ticket_metadata.json") as f: t_meta = json.load(f)
    with open("policy_metadata.json") as f: p_meta = json.load(f)
    return groq_client, vectorizer, clf, embedder, t_index, p_index, t_meta, p_meta

groq_client, vectorizer, clf, embedder, ticket_index, policy_index, ticket_metadata, policy_metadata = load_artifacts()

# ==============================================================
# AGENT (identical logic to m6, no print statements)
# ==============================================================

CRITICAL_KEYWORDS = ["breach", "hacked", "unauthorized access", "data leak", "outage", "down for everyone"]
HIGH_KEYWORDS     = ["charged twice", "charged three times", "double charge", "cannot access", "urgent", "immediately"]

def infer_priority(text):
    t = text.lower()
    if any(k in t for k in CRITICAL_KEYWORDS): return "critical"
    if any(k in t for k in HIGH_KEYWORDS):     return "high"
    if "?" in text and len(text) < 200:        return "low"
    return "medium"

class AgentState(TypedDict):
    ticket_body:             str
    queue:                   Optional[str]
    priority:                Optional[str]
    confidence:              Optional[float]
    route:                   Optional[str]
    retrieved_context:       List[dict]
    generated_reply:         Optional[str]
    needs_clarification:     bool
    clarification_question:  Optional[str]
    log:                     dict

def classify_ticket(state):
    body  = state["ticket_body"]
    vec   = vectorizer.transform([body])
    queue = clf.predict(vec)[0]
    conf  = float(clf.predict_proba(vec).max())
    priority = infer_priority(body)
    return {**state, "queue": queue, "priority": priority, "confidence": round(conf, 3),
            "log": {**state["log"], "queue": queue, "priority": priority, "confidence": round(conf, 3)}}

def route_decision(state):
    priority = state["priority"]
    body     = state["ticket_body"]
    body_low = body.lower()
    if priority == "critical":
        route = "escalate"
    elif any(k in body_low for k in [
        "how long", "what is your policy", "refund policy", "sla", "response time",
        "within how many days", "what are the terms", "policy", "procedure",
        "payment plans", "billing options", "subscription plans", "what are your"
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
             "support agent immediately. You will be contacted within 1 hour via a secure channel. "
             "Please do not share sensitive credentials in this ticket.")
    return {**state, "generated_reply": reply, "log": {**state["log"], "escalated": True}}

def ask_clarifying_question(state):
    question = ("Thank you for reaching out. To help you as quickly as possible, could you "
                "please provide more details? Which product or service is affected, what error "
                "you're seeing, and when the issue started.")
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
        prompt = (f"You are a professional customer support agent.\n"
                  f"Use ONLY these past resolved tickets as reference to draft a reply.\n"
                  f"Do not make up information not in the context.\n\n"
                  f"PAST TICKETS:\n{context_block}\n\n"
                  f"NEW TICKET:\n{body}\n\nDraft a professional reply:")
    else:
        context_block = "\n\n".join([f"Policy ({c['filename']}):\n{c['content']}" for c in context])
        prompt = (f"You are a professional customer support agent.\n"
                  f"Use ONLY this policy document to answer the customer.\n"
                  f"Do not make up information not in the policy.\n\n"
                  f"POLICY:\n{context_block}\n\n"
                  f"TICKET:\n{body}\n\nDraft a professional reply:")
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

@st.cache_resource
def build_agent():
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
    return graph.compile()

agent = build_agent()

# ==============================================================
# UI
# ==============================================================

st.title("🎫 Ticket Triage Agent")
st.caption("Paste a support ticket below. The agent classifies it, decides how to handle it, retrieves relevant context, and drafts a reply — showing its reasoning at every step.")

st.divider()

# ---- Example tickets ----
with st.expander("Try an example ticket"):
    examples = {
        "Billing complaint": "I was charged twice for my subscription this month and need this fixed immediately.",
        "Policy question":   "What is your refund policy? How long do I have to request one?",
        "Security incident": "We think our account was hacked. Someone accessed our data without authorization.",
        "Vague ticket":      "Hi, it's not working.",
    }
    selected = st.selectbox("Choose an example", list(examples.keys()))
    if st.button("Load example"):
        st.session_state["ticket_input"] = examples[selected]

# ---- Ticket input ----
ticket_body = st.text_area(
    "Support ticket",
    value=st.session_state.get("ticket_input", ""),
    height=140,
    placeholder="Paste the customer's support ticket here...",
    label_visibility="collapsed"
)

run = st.button("Triage Ticket", type="primary", use_container_width=True)

# ---- Run agent ----
if run:
    if not ticket_body.strip():
        st.warning("Please paste a ticket first.")
    else:
        with st.spinner("Triaging..."):
            initial_state: AgentState = {
                "ticket_body": ticket_body, "queue": None, "priority": None,
                "confidence": None, "route": None, "retrieved_context": [],
                "generated_reply": None, "needs_clarification": False,
                "clarification_question": None, "log": {}
            }
            result = agent.invoke(initial_state)

        st.divider()

        # ---- Step 1: Classification ----
        priority_class = f"badge-priority-{result['priority']}"
        st.markdown("**Step 1 — Classification**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Queue</div>
                <div class="step-value">{result['queue']}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Priority</div>
                <div class="step-value">
                    <span class="badge {priority_class}">{result['priority'].upper()}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Confidence</div>
                <div class="step-value">{result['confidence']:.0%}</div>
            </div>""", unsafe_allow_html=True)

        # ---- Step 2: Routing ----
        route = result["route"]
        route_badge_class = {
            "similar_tickets": "badge-similar",
            "policy_kb":       "badge-policy",
            "escalate":        "badge-escalate",
            "ask_clarifying":  "badge-clarify"
        }.get(route, "badge-similar")

        route_explanation = {
            "similar_tickets": "Routed to similar-ticket retrieval — specific complaint with enough detail to match past resolutions.",
            "policy_kb":       "Routed to policy KB — ticket asks about rules, timelines, or procedures.",
            "escalate":        "Escalated immediately — security incident or critical priority detected.",
            "ask_clarifying":  "Requesting more information — ticket is too vague to retrieve relevant context."
        }.get(route, "")

        st.markdown("**Step 2 — Routing Decision**")
        st.markdown(f"""
        <div class="step-card">
            <div class="step-label">Route taken</div>
            <div class="step-value" style="margin-bottom:0.4rem">
                <span class="badge {route_badge_class}">{route.replace('_', ' ')}</span>
            </div>
            <div style="font-size:0.85rem; color:#6b7280; margin-top:0.4rem">{route_explanation}</div>
        </div>""", unsafe_allow_html=True)

        # ---- Step 3: Retrieved context ----
        if result["retrieved_context"]:
            st.markdown("**Step 3 — Retrieved Context**")
            ctx = result["retrieved_context"]

            if ctx[0]["type"] == "similar_ticket":
                top = ctx[0]
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-label">Top similar ticket (similarity: {top['score']:.3f})</div>
                    <div class="context-box"><b>Customer:</b> {top['body'][:250]}...
<b>Agent reply:</b> {top['answer'][:250]}...</div>
                </div>""", unsafe_allow_html=True)
            else:
                top = ctx[0]
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-label">Policy document retrieved: {top['filename']} (similarity: {top['score']:.3f})</div>
                    <div class="context-box">{top['content'][:400]}...</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("**Step 3 — Retrieved Context**")
            st.markdown("""
            <div class="step-card">
                <div class="step-label">Retrieval</div>
                <div class="step-value" style="color:#6b7280">Skipped — ticket was escalated or sent for clarification.</div>
            </div>""", unsafe_allow_html=True)

        # ---- Step 4: Generated reply ----
        st.markdown("**Step 4 — Generated Reply**")
        st.markdown(f"""
        <div class="step-card">
            <div class="step-label">Draft response</div>
            <div class="reply-box">{result['generated_reply']}</div>
        </div>""", unsafe_allow_html=True)

        # Copy button
        st.code(result["generated_reply"], language=None)