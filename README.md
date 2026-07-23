# 🎫 Intelligent Ticket Triage & Resolution Agent

An end-to-end AI system that automatically classifies incoming support tickets, routes them through the appropriate handling path, retrieves grounded context, and drafts professional replies — with full reasoning transparency at every step.

**[Live Demo →](https://ticket-triage-agent-7wgesln3bwugqxn2ueksn2.streamlit.app/)** &nbsp;|&nbsp; **[GitHub →](https://github.com/tejaravi1718-collab/ticket-triage-agent)**



## What it does

When a support ticket comes in, the agent:

1. **Classifies** it by category (7 queues) and urgency (critical / high / medium / low) using a trained ML model
2. **Routes** it through one of four decision branches based on classifier output and routing rules
3. **Retrieves** relevant context — either similar past resolved tickets or a policy document
4. **Generates** a grounded draft reply using an LLM, citing only retrieved context (no hallucination)
5. **Logs** every decision for evaluation and auditability

All four steps are visible in the UI — not a black box.

---

## Architecture

```
New ticket
    │
    ▼
┌─────────────────────┐
│   Classify Ticket   │  ← TF-IDF + Logistic Regression (trained ML model)
│  queue + priority   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Route Decision    │  ← Rule-based routing on classifier output + keywords
└─────────────────────┘
    │
    ├── critical/security ──────────→ [ Escalate ]
    │
    ├── policy question ────────────→ [ Policy KB Retrieval ]
    │                                        │
    ├── routine complaint ──────────→ [ Similar Ticket Retrieval ]
    │                                        │
    └── too vague ──────────────────→ [ Ask Clarifying Question ]
                                             │
                                             ▼
                                    [ Generate Reply ]
                                    (grounded in retrieved context)
                                             │
                                             ▼
                                    [ Log Decision ]
                                    (for evaluation)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| ML Classifier | scikit-learn (TF-IDF + Logistic Regression) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (two indexes: similar tickets + policy KB) |
| Agent orchestration | LangGraph (StateGraph with conditional routing) |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Evaluation Results

Evaluated against a 50-ticket hand-labeled test set:

| Metric | Score |
|---|---|
| **Routing accuracy** | **84%** (42/50 correct) |
| Classifier macro-F1 | 0.54 (7-class) |
| Classifier accuracy | 56% |

**Routing breakdown by branch:**

| Branch | Accuracy |
|---|---|
| similar_tickets | 90.2% (37/41) |
| escalate | 100.0% (2/2) |
| ask_clarifying | 66.7% (2/3) |
| policy_kb | 25.0% (1/4) |

**Documented failure modes:**
- `policy_kb` under-triggered: keyword list too narrow, misses natural language variations ("payment plans", "billing options")
- `escalate` over-triggered: "outage" keyword incorrectly escalates service disruptions alongside security incidents
- `ask_clarifying` under-triggered: length threshold alone insufficient for vagueness detection

---

## Dataset

**[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)** — synthetic IT support ticket dataset with category and priority labels.

- 28,260 English tickets after filtering
- 10 original queue categories → merged to 7 based on confusion matrix evidence
- Near-duplicate removal applied before index construction
- ⚠️ Synthetic dataset (AI-generated) — disclosed honestly

**Why category merging?** Embedding-based classification (all-MiniLM-L6-v2) underperformed TF-IDF (31% vs 48% accuracy), revealing that distinguishing signals in this dataset are vocabulary-driven rather than semantics-driven. Confusion matrix analysis showed IT Support and Technical Support were nearly indistinguishable, leading to evidence-based merging from 10 → 7 categories.

---

## Project Structure

```
ticket-triage-agent/
├── app.py                      # Streamlit UI (main entry point)
├── requirements.txt
├── policy_kb/                  # 6 policy documents for policy retrieval
│   ├── billing_disputes.md
│   ├── escalation_criteria.md
│   ├── general_support_process.md
│   ├── refunds_and_returns.md
│   ├── security_incident_handling.md
│   └── sla_response_times.md
├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
├── queue_classifier.pkl        # Trained Logistic Regression classifier
├── ticket_index.faiss          # FAISS index: 10k similar tickets
├── ticket_metadata.json        # Ticket bodies + answers for retrieval
├── policy_index.faiss          # FAISS index: 6 policy documents
├── policy_metadata.json        # Policy document content
├── m1_explore_data.py          # Data exploration + EDA
├── m2_baseline_classifier.py   # TF-IDF baseline (48% accuracy)
├── m3_embedding_classifier.py  # Embedding comparison (31% accuracy)
├── m3b_merged_classifier.py    # Merged categories (56% accuracy)
├── save_model.py               # Save trained classifier to disk
├── m4_build_indexes.py         # Build both FAISS indexes
├── m5_rag_generation.py        # RAG generation (both modes)
├── m6_agent.py                 # LangGraph agent (4-branch routing)
├── m7_create_eval_set.py       # Create hand-labeled eval CSV
└── m7_eval.py                  # Run evaluation + compute metrics
```

---

## How to Run Locally

```bash
git clone https://github.com/tejaravi1718-collab/ticket-triage-agent
cd ticket-triage-agent
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

To rebuild from scratch (retrain classifier + rebuild indexes):
```bash
python m3b_merged_classifier.py   # train classifier
python save_model.py              # save to disk
python m4_build_indexes.py        # build FAISS indexes
```

---

## Key Design Decisions

**Why TF-IDF over embeddings for classification?**
Embedding-based classification (all-MiniLM-L6-v2) achieved only 31% accuracy vs TF-IDF's 48% on this dataset. Investigation showed the distinguishing signal between ticket categories is vocabulary-specific (exact domain terms) rather than semantic — a case where a simpler model genuinely outperforms a more complex one.

**Why two retrieval modes?**
Policy questions require authoritative, consistent answers from a controlled source. Routine complaints are better served by past resolved tickets. A single retrieval mode would either hallucinate policy details or return generic policy text for specific complaints.

**Why keyword-based routing instead of LLM routing?**
Keyword routing is fast, auditable, and doesn't consume LLM tokens for every ticket. The routing logic is explicit and debuggable — when it fails, you can see exactly why and fix the rule. LLM-based routing would add latency and cost to every request.