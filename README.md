# 🤖 RAG System: Retrieval-Augmented Generation from Scratch

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of a Retrieval-Augmented Generation (RAG) system built from scratch in Python. This project demonstrates how to build an intelligent document query system that can process multiple file formats, generate embeddings, and provide contextual answers to user questions.

## 🌟 Features

- **Multi-Format Support**: Process PDF, TXT, and DOCX files seamlessly
- **Intelligent Chunking**: Smart text segmentation with configurable overlap
- **Semantic Search**: Uses sentence transformers for accurate document retrieval
- **Vector Database**: Simple in-memory vector storage with cosine similarity search
- **Flexible Response Generation**: Support for multiple LLM backends (OpenAI, Ollama, or simple retrieval)
- **Persistent Storage**: Save and load processed documents for faster subsequent queries
- **Easy to Extend**: Clean, modular architecture for customization

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install sentence-transformers PyPDF2 python-docx transformers torch numpy
```

### Optional Dependencies

For OpenAI integration:
```bash
pip install openai
```

For Ollama (local LLM):
```bash
# Install Ollama from https://ollama.com
ollama pull llama2
```

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

### 2. Prepare Your Documents

Create a `documents` folder and add your files:
```
rag-system/
├── rag_system.py
└── documents/
    ├── document1.pdf
    ├── document2.txt
    └── document3.docx
```

### 3. Run the System

```python
from rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(response_method='simple')

# Add your documents
file_paths = [
    'documents/document1.pdf',
    'documents/document2.txt',
    'documents/document3.docx'
]
rag.add_documents(file_paths, chunk_size=500)

# Save the database for future use
rag.save_database('my_knowledge_base.json')

# Query the system
response = rag.query("What are the main topics covered?", top_k=3)
print(response)
```

## 💡 Usage Examples

### Basic Document Processing

```python
# Initialize the system
rag = RAGSystem(
    embedding_model='all-MiniLM-L6-v2',
    response_method='simple'
)

# Add documents
rag.add_documents(['resume.pdf', 'research_paper.pdf'], chunk_size=500)

# Query
answer = rag.query("What are the key findings?", top_k=3)
print(answer)
```

### Using OpenAI for Response Generation

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

rag = RAGSystem(response_method='openai')
rag.add_documents(['documents/file.pdf'])

# Get AI-generated answers
response = rag.query("Summarize the main points", top_k=5)
print(response)
```

### Using Ollama (Free, Local)

```python
# Make sure Ollama is running: ollama serve
rag = RAGSystem(response_method='ollama')
rag.add_documents(['documents/file.pdf'])

response = rag.query("Explain this concept", top_k=3)
print(response)
```

### Loading Pre-processed Database

```python
# Load existing database (skip re-processing)
rag = RAGSystem()
rag.load_database('my_knowledge_base.json')

# Start querying immediately
response = rag.query("Your question here?")
```

## 🏗️ Architecture

The RAG system consists of five main components:

### 1. **DocumentLoader**
- Supports PDF, TXT, and DOCX formats
- Handles encoding and extraction
- Easy to extend for additional formats

### 2. **TextChunker**
- Splits documents into manageable chunks
- Configurable chunk size and overlap
- Preserves context across chunks

### 3. **EmbeddingGenerator**
- Uses Sentence Transformers for semantic embeddings
- Supports multiple embedding models
- Efficient batch processing

### 4. **SimpleVectorDB**
- In-memory vector storage
- Cosine similarity search
- Save/load functionality
- Metadata support

### 5. **ResponseGenerator**
- Multiple backend support (Simple, OpenAI, Ollama)
- Contextual answer generation
- Customizable prompting

## ⚙️ Configuration

### Embedding Models

Choose from various pre-trained models:

```python
# Fast and efficient (default)
rag = RAGSystem(embedding_model='all-MiniLM-L6-v2')

# Better quality, slower
rag = RAGSystem(embedding_model='all-mpnet-base-v2')

# Optimized for Q&A
rag = RAGSystem(embedding_model='multi-qa-mpnet-base-dot-v1')
```

### Chunking Strategy

```python
# Adjust chunk size and overlap
rag.add_documents(
    file_paths,
    chunk_size=500,  # words per chunk
)
```

### Response Methods

```python
# Simple retrieval (no LLM)
rag = RAGSystem(response_method='simple')

# OpenAI GPT
rag = RAGSystem(response_method='openai')

# Local Ollama
rag = RAGSystem(response_method='ollama')
```

## 🔧 Advanced Features

### Custom Metadata

```python
# Add metadata to track document sources
metadata = [
    {'source': 'research_paper.pdf', 'author': 'John Doe', 'year': 2024},
    {'source': 'notes.txt', 'category': 'personal'}
]
```

### Fine-tune Retrieval

```python
# Retrieve more or fewer chunks
response = rag.query("Your question?", top_k=5)  # Get top 5 results
```

### Export Results

```python
# Save database for later use
rag.save_database('knowledge_base.json')

# Load in another session
new_rag = RAGSystem()
new_rag.load_database('knowledge_base.json')
```

## 📊 Performance Tips

1. **Chunk Size**: Start with 500 words, adjust based on document type
2. **Overlap**: 10% overlap helps maintain context
3. **Top K**: Use 3-5 chunks for most queries
4. **Embedding Model**: Balance speed vs accuracy based on use case
5. **Database Caching**: Save processed databases to avoid re-computation

## 🛠️ Project Structure

```
rag-system/
├── rag_system.py          # Main implementation
├── documents/             # Your document files
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── examples/             # Usage examples
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- [ ] Add support for more file formats (Excel, CSV, HTML)
- [ ] Implement hybrid search (dense + sparse)
- [ ] Add re-ranking capabilities
- [ ] Build web UI with Streamlit/Gradio
- [ ] Add support for more vector databases (Pinecone, Weaviate)
- [ ] Implement query rewriting
- [ ] Add evaluation metrics

## 📝 Use Cases

- **Research Assistant**: Query academic papers and research documents
- **Customer Support**: Build knowledge bases from documentation
- **Personal Knowledge Management**: Search through notes and documents
- **Legal/Compliance**: Find relevant regulations and policies
- **Healthcare**: Query medical literature and patient records
- **Code Documentation**: Search through technical documentation

## 🐛 Troubleshooting

### Common Issues

**Import Error: No module named 'sentence_transformers'**
```bash
pip install sentence-transformers
```

**PDF Reading Error**
```bash
pip install PyPDF2
```

**CUDA/GPU Issues**
```bash
# For CPU-only installation
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**File Path Issues (Windows)**
```python
# Use raw strings
file_path = r"C:\Users\Documents\file.pdf"
# Or forward slashes
file_path = "C:/Users/Documents/file.pdf"
```

## 📚 Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [RAG Concepts Explained](https://arxiv.org/abs/2005.11401)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Ollama Documentation](https://ollama.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence Transformers team for the embedding models
- OpenAI for GPT models
- Ollama team for local LLM support
- The open-source community

## 📧 Contact

**Sujith Madhusudhana**
- Email: sujithh.sj@gmail.com
- LinkedIn: [linkedin.com/in/sujith-madhusudhana](https://linkedin.com/in/sujith-madhusudhana)
- GitHub: [@yourusername](https://github.com/yourusername)

---

⭐ If you found this project helpful, please consider giving it a star!

**Built with ❤️ by Sujith Madhusudhana**
