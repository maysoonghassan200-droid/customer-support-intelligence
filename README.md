# Customer Support Ticket Intelligence & RAG Assistant

An AI-powered customer support system that automatically classifies support tickets, retrieves relevant historical solutions, and generates context-aware responses using Machine Learning, Retrieval-Augmented Generation (RAG), and Gemini.

## Overview

The project combines two main AI pipelines:

1. **Ticket Intelligence**
   - Predicts the appropriate support queue.
   - Predicts ticket priority.
   - Uses TF-IDF and Logistic Regression models.

2. **RAG Support Assistant**
   - Searches historical support answers using semantic embeddings.
   - Uses FAISS for vector similarity search.
   - Uses Gemini to generate a final support response.
   - Maintains conversation memory for follow-up messages.

## System Architecture

```text
Customer Ticket
      |
      v
React Dashboard
      |
      v
FastAPI Backend
      |
      +----------------------+
      |                      |
      v                      v
ML Ticket Triage        RAG Pipeline
      |                      |
      v                      v
Queue + Priority       SentenceTransformer
Prediction                   |
                             v
                         FAISS Search
                             |
                             v
                    Relevant Historical
                         Answers
                             |
                             v
                           Gemini
                             |
                             v
                    Support Response
```

## Dataset

The project uses a cleaned sample of **8,000 English customer support tickets**.

The dataset contains information such as:

- Ticket subject
- Ticket body
- Historical answer
- Queue
- Priority
- Ticket type
- Tags

After chunking the historical answers, the RAG knowledge base contains **8,320 chunks**.

## Machine Learning

### Queue Classification

Baseline majority classifier:

- Accuracy: **28.63%**
- Macro F1: **4.45%**

TF-IDF + Logistic Regression:

- Accuracy: **43.81%**
- Macro F1: **27.93%**

DistilBERT experiment:

- Accuracy: **41.13%**
- Macro F1: **27.55%**

TF-IDF + Logistic Regression was selected for the final API because it achieved slightly better queue accuracy while remaining lightweight and fast.

### Priority Classification

Baseline majority classifier:

- Accuracy: **41.69%**
- Macro F1: **19.61%**

TF-IDF + Logistic Regression:

- Accuracy: **53.56%**
- Macro F1: **45.70%**

## Retrieval-Augmented Generation

The RAG pipeline uses:

- `all-MiniLM-L6-v2` SentenceTransformer
- 384-dimensional embeddings
- FAISS `IndexFlatL2`
- 8,320 knowledge-base chunks
- Gemini for response generation

The pipeline retrieves relevant historical answers and provides them as context to Gemini before generating the final response.

## Conversation Memory

The assistant maintains session-based conversation memory.

For example:

```text
User: My payment keeps failing.

User: It happened yesterday.
```

The system understands that the second message refers to the payment problem from the previous message and uses the conversation context for both triage and response generation.

## API

The backend is built with **FastAPI**.

### Endpoints

#### Home

```http
GET /
```

Confirms that the Customer Support API is running.

#### System Health

```http
GET /health
```

Returns:

- API connection status
- Vector store status
- Number of FAISS vectors
- Number of knowledge-base chunks

#### Ticket Triage

```http
POST /triage
```

Predicts:

- Support queue
- Ticket priority

#### Support Assistant

```http
POST /chat
```

Runs the complete support pipeline:

```text
Conversation Context
        ↓
Ticket Triage
        ↓
FAISS Knowledge Search
        ↓
Gemini Generation
        ↓
Final Support Response
```

#### System Statistics

```http
GET /stats
```

Returns project information including:

- Total tickets
- Knowledge-base chunks
- FAISS vectors
- Number of support queues
- Queue classification model
- Priority classification model
- Embedding model
- LLM

#### Ticket Preview

```http
GET /tickets?limit=10
```

Returns a preview of ticket information including:

- Ticket ID
- Queue
- Priority

The requested limit is restricted to a range of **1–100 tickets**.

### Swagger Documentation

Interactive API documentation is available through FastAPI Swagger UI at:

```text
http://localhost:8010/docs
```

## Frontend

The frontend is built with **React + Vite**.

The dashboard provides:

- Support assistant chat
- Predicted queue
- Predicted priority
- Knowledge-base information
- Conversation memory
- API connection status
- System pipeline visualization

## Project Structure

```text
customer-support-intelligence/
│
├── main.py
├── requirements.txt
├── queue_model.joblib
├── priority_model.joblib
├── rag_chunks.csv
├── tickets_faiss.index
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
└── frontend/
    ├── src/
    ├── public/
    ├── package.json
    ├── package-lock.json
    ├── vite.config.js
    └── Dockerfile
```

## Run Locally

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gemini

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start the Backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8010
```

### 4. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

## Docker

The entire application can be started using Docker Compose:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8010`
- Swagger API: `http://localhost:8010/docs`

## Technologies

**Machine Learning:** Python, scikit-learn, TF-IDF, Logistic Regression, DistilBERT

**RAG:** SentenceTransformers, FAISS, Gemini

**Backend:** FastAPI, Uvicorn

**Frontend:** React, Vite

**Deployment & Version Control:** Docker, Docker Compose, Git, GitHub

## Security

API keys and environment secrets are not committed to the repository.

The `.env` file is excluded through `.gitignore`, while `.env.example` documents the required environment variables without exposing credentials.

## Key Features

- Automatic ticket queue classification
- Automatic priority prediction
- Semantic retrieval from historical support tickets
- Gemini-powered response generation
- Conversation-aware follow-up handling
- System health monitoring
- Project statistics endpoint
- Ticket preview endpoint
- FastAPI REST API
- Interactive Swagger documentation
- React dashboard
- Dockerized application
- Secure environment variable management

## Author

**Maysoon Alshaikh Saleh**