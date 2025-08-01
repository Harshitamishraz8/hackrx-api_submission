# HackRx Document Q&A API

A FastAPI-based document question-answering system that processes PDF documents and answers questions using RAG (Retrieval-Augmented Generation) with Pinecone vector database and Groq LLM.

## Features

- 📄 PDF document processing from URLs (including Google Drive)
- 🔍 Vector-based semantic search using Pinecone
- 🤖 AI-powered question answering with Groq LLM
- 🔐 Bearer token authentication
- ⚡ Fast response times optimized for competition requirements

## API Endpoint

### POST `/hackrx/run`

Process a PDF document and answer questions about its content.

**Headers:**
```
Authorization: Bearer hackrx-secret-token
Content-Type: application/json
```

**Request Body:**
```json
{
  "documents": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
  "questions": [
    "What is this document about?",
    "Who is eligible for this program?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "This document describes...",
    "The eligible candidates are..."
  ]
}
```

## Environment Variables

Set these in your Railway deployment:

- `GROQ_API_KEY`: Your Groq API key
- `PINECONE_API_KEY`: Your Pinecone API key  
- `PINECONE_ENVIRONMENT`: Pinecone environment (e.g., "gcp-starter")
- `PINECONE_INDEX`: Pinecone index name (e.g., "hackrx-index")
- `HACKRX_API_TOKEN`: Authentication token (default: "hackrx-secret-token")
- `EMBEDDING_MODEL`: Sentence transformer model (default: "all-MiniLM-L6-v2")

## Testing

Use Postman or curl to test:

```bash
curl -X POST "https://your-railway-url.railway.app/hackrx/run" \
  -H "Authorization: Bearer hackrx-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://drive.google.com/uc?export=download&id=1kneuhOyIgoJphUZkncsfDXOOEIEw2xe1",
    "questions": ["What is this document about?"]
  }'
```

## Architecture

1. **PDF Processing**: Downloads and extracts text using pdfminer
2. **Text Chunking**: Splits documents into overlapping chunks
3. **Vector Storage**: Generates embeddings and stores in Pinecone
4. **Query Processing**: Finds relevant chunks using semantic search
5. **Answer Generation**: Uses Groq LLM to generate contextual answers

## Deployment

This API is configured for Railway deployment with automatic builds and restarts.