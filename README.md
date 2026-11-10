# 🛡️ CyberShield AI – Intelligent Cybersecurity RAG Assistant

**CyberShield AI** is an AI-powered cybersecurity assistant built using **Retrieval-Augmented Generation (RAG)**. It enables users to upload cybersecurity documents and ask questions about their content through semantic search and Google's Gemini AI.

The system retrieves relevant information from uploaded documents and uses it as context to generate informative, context-aware answers with source references.

## 🚀 Key Features

### 📄 Document Management
- Upload PDF, TXT, and DOCX cybersecurity documents.
- Extract text from different document formats.
- Automatically split documents into smaller overlapping chunks.
- View uploaded documents and document statistics.
- Delete previously uploaded documents.

### 🧠 Retrieval-Augmented Generation (RAG)
- Generate semantic embeddings using Sentence Transformers.
- Store document embeddings in ChromaDB.
- Perform semantic similarity searches.
- Retrieve relevant document sections based on user questions.
- Generate context-aware answers using Google Gemini.
- Display source references to support AI-generated answers.
- Avoid generating unsupported answers when relevant knowledge is unavailable.

### 💻 User Interface
- React-based interactive dashboard.
- Document upload and management interface.
- AI-powered cybersecurity chat interface.
- Document statistics and search results.
- Source information displayed alongside AI responses.

### ⚙️ Backend and API
- RESTful APIs developed using FastAPI.
- Interactive Swagger API documentation.
- Input validation and file-size restrictions.
- Centralized configuration using environment variables.
- Error handling for document processing and AI generation.

## 🏗️ System Architecture

### Document Ingestion Pipeline

```text
             Cybersecurity Documents
                  PDF / TXT / DOCX
                         |
                         v
                  Document Upload
                         |
                         v
                   Text Extraction
                         |
                         v
                    Text Chunking
                         |
                         v
                Sentence Transformers
                     Embeddings
                         |
                         v
                      ChromaDB
                  Vector Storage
```

### RAG Question-Answering Pipeline

```text
                     User Question
                          |
                          v
                Generate Query Embedding
                          |
                          v
               Semantic Similarity Search
                          |
                          v
                Retrieve Relevant Context
                          |
                          v
                   Google Gemini AI
                          |
                          v
                Context-Aware Response
                          |
                          v
                 Source References
```

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Backend development |
| FastAPI | REST API framework |
| React | Frontend development |
| Vite | Frontend build tooling |
| JavaScript | Frontend functionality |
| Google Gemini | AI response generation |
| Google GenAI SDK | Gemini integration |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pydantic Settings | Environment configuration |
| Pytest | Automated testing |
| Pytest-Cov | Code coverage |

## 📂 Project Structure

```text
CyberShield-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   └── documents.py
│   │   │
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── ingestion.py
│   │   │   └── generation.py
│   │   │
│   │   ├── services/
│   │   │   └── chat_service.py
│   │   │
│   │   ├── config.py
│   │   └── main.py
│   │
│   ├── data/
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── ...
│   │
│   ├── package.json
│   └── ...
│
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/M-Abubakar24/CyberShield-AI.git
cd CyberShield-AI
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file inside the `backend` directory.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash
```

Replace the API key with your own Google Gemini API key.

**Important:** Never commit your actual `.env` file or API key to GitHub.

### 4. Run the Backend

From the backend directory:

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 5. Frontend Setup

Open a second terminal and navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Backend health check |
| POST | `/api/documents/upload` | Upload a document |
| GET | `/api/documents/` | List uploaded documents |
| GET | `/api/documents/stats` | Retrieve document statistics |
| POST | `/api/documents/search` | Search document content |
| DELETE | `/api/documents/{document_id}` | Delete a document |
| POST | `/api/chat/ask` | Ask cybersecurity questions |

## 🧪 Testing and Code Coverage

CyberShield AI includes a comprehensive automated backend test suite using **Pytest** and **Pytest-Cov**.

### Latest Test Results

| Metric | Result |
|---|---:|
| Total tests | 88 |
| Passed | 88 |
| Failed | 0 |
| Statements measured | 300 |
| Backend code coverage | 100% |

**All 88 automated tests passed successfully.**

### Run Tests

From the backend directory:

```bash
python -m pytest -v
```

Run tests with coverage:

```bash
python -m pytest --cov=app --cov-report=term-missing -v
```

## 🔐 Security Considerations

CyberShield AI is designed as a defensive cybersecurity knowledge assistant.

Current protections include:
- File upload size restrictions.
- Supported file format validation.
- Empty document validation.
- Input validation through Pydantic.
- Error handling for document processing and AI generation.
- Environment-based API key configuration.
- Retrieval-grounded AI responses.

**Note:** The current prototype is intended for local development. Authentication, user-specific document access, production upload hardening, and deployment security should be implemented and reviewed before exposing it publicly.

## 🗺️ Future Improvements

- [ ] JWT authentication and user authorization.
- [ ] User-specific document ownership and access control.
- [ ] Advanced document filtering and pagination.
- [ ] Improved PDF processing and OCR support.
- [ ] Chat history and conversation management.
- [ ] Streaming AI responses.
- [ ] API rate limiting.
- [ ] Security headers and production CORS configuration.
- [ ] Docker containerization.
- [ ] PostgreSQL integration.
- [ ] Cloud deployment.

## 🎯 Project Objective

The objective of CyberShield AI is to combine **Artificial Intelligence, Retrieval-Augmented Generation, and Cybersecurity Knowledge Management** into one practical application.

It demonstrates the implementation of a complete RAG pipeline, semantic document retrieval, vector database integration, AI-powered question answering, REST API development, and automated backend testing.

---

## 👨‍💻 Author

**Muhammad Abubakar**

Software Engineering Graduate | Backend Development | AI & Cybersecurity

- GitHub: [M-Abubakar24](https://github.com/M-Abubakar24)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/mabubakar2492/)
- Email: m.abubakar2496@gmail.com

---
