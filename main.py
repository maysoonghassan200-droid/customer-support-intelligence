
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import joblib
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

# -----------------------------
# Setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(
    title="Customer Support Ticket Intelligence API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models and RAG files
# -----------------------------

queue_model = joblib.load(
    os.path.join(BASE_DIR, "queue_model.joblib")
)

priority_model = joblib.load(
    os.path.join(BASE_DIR, "priority_model.joblib")
)

chunks_df = pd.read_csv(
    os.path.join(BASE_DIR, "rag_chunks.csv")
)

index = faiss.read_index(
    os.path.join(BASE_DIR, "tickets_faiss.index")
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

conversation_memory = {}

# -----------------------------
# Request models
# -----------------------------

class TriageRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


# -----------------------------
# Helper functions
# -----------------------------

def clean_answer_text(text):
    replacements = {
        "<name>": "there",
        "<acc_num>": "account number",
        "<tel_num>": "phone number"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def add_to_memory(session_id, role, message):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    conversation_memory[session_id].append({
        "role": role,
        "message": message
    })


def get_memory(session_id):
    return conversation_memory.get(session_id, [])


def ticket_triage(ticket_text):
    queue = queue_model.predict([ticket_text])[0]
    priority = priority_model.predict([ticket_text])[0]

    return {
        "queue": queue,
        "priority": priority
    }


def retrieve_answers(query, k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):
        row = chunks_df.iloc[idx]

        results.append({
            "ticket_id": int(row["ticket_id"]),
            "queue": row["queue"],
            "priority": row["priority"],
            "answer": clean_answer_text(row["text"]),
            "distance": float(distance)
        })

    return results


def support_agent(session_id, user_message):

    previous_history = get_memory(session_id)

    previous_user_messages = [
        item["message"]
        for item in previous_history
        if item["role"] == "user"
    ]

    tool_text = " ".join(
        previous_user_messages + [user_message]
    )

    add_to_memory(
        session_id,
        "user",
        user_message
    )

    triage = ticket_triage(tool_text)

    kb_results = retrieve_answers(
        tool_text,
        k=3
    )

    context_parts = []

    for i, result in enumerate(
        kb_results,
        start=1
    ):
        context_parts.append(
            f"""Source {i}:
Queue: {result['queue']}
Priority: {result['priority']}
Answer: {result['answer']}"""
        )

    context = "\n\n".join(context_parts)

    history = get_memory(session_id)

    history_text = "\n".join(
        f"{item['role']}: {item['message']}"
        for item in history
    )

    prompt = f"""
You are a customer support assistant.

Predicted queue:
{triage['queue']}

Predicted priority:
{triage['priority']}

Conversation history:
{history_text}

Knowledge base:
{context}

Answer the user's latest message using the knowledge base
and conversation history.

Do not invent information.
Do not use placeholders such as <name>, <acc_num>, or <tel_num>.
If more information is needed, ask clearly.

Latest user message:
{user_message}

Answer:
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = clean_answer_text(
        response.text
    )

    add_to_memory(
        session_id,
        "assistant",
        answer
    )

    return {
        "queue": triage["queue"],
        "priority": triage["priority"],
        "answer": answer
    }


# -----------------------------
# API endpoints
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Support API is running"
    }


@app.post("/triage")
def triage_endpoint(request: TriageRequest):
    return ticket_triage(
        request.text
    )


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return support_agent(
        request.session_id,
        request.message
    )
