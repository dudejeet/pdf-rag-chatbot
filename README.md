# AI-Powered PDF Q&A Chatbot

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content.

The system:
- Extracts text from PDF files
- Splits content into chunks
- Generates vector embeddings
- Stores embeddings in a FAISS vector database
- Retrieves relevant context
- Uses an LLM to generate accurate responses

## Technologies Used

- Python
- LangChain
- OpenAI
- FAISS
- Streamlit
- PyPDF

## Features

- PDF document upload
- Semantic search
- Retrieval-Augmented Generation (RAG)
- Context-aware question answering
- Interactive Streamlit interface

## Future Improvements

- Multi-document support
- Conversation memory
- Citation support
- Local LLM integration
