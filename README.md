# 🛡️ CyberShield AI – Intelligent Cybersecurity RAG Assistant

CyberShield AI is an AI-powered cybersecurity assistant built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload cybersecurity documents and ask questions about their content using semantic search and Google's Gemini AI.

The system retrieves relevant information from uploaded documents and uses it as context to generate informative, context-aware answers.

## 🚀 Key Features

- **Document Upload:** Upload PDF, TXT, and DOCX cybersecurity documents.
- **Text Extraction:** Extract content from uploaded documents.
- **Intelligent Chunking:** Divide documents into smaller overlapping text chunks.
- **Semantic Embeddings:** Generate vector embeddings using Sentence Transformers.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic Search:** Find relevant document sections based on user questions.
- **AI-Powered Answers:** Generate contextual responses using Google Gemini.
- **REST API:** Backend APIs built with FastAPI.
- **API Documentation:** Interactive Swagger documentation.

## 🏗️ Architecture

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
                    (Embeddings)
                           |
                           v
                       ChromaDB
                    (Vector Storage)


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
                  Contextual AI Response
```

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| FastAPI | REST API framework |
| Google Gemini | AI response generation |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pydantic Settings | Environment configuration |
| Pytest | Testing |

## 📂 Project Structure

```text
CyberShield-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   └── documents.py
│   │   │
│   │   ├── rag/
│   │   │   ├── chunking.py
│   │   │   ├
