# 🤖 RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, LangChain, FAISS, OpenAI API, and OpenRouter. Upload documents and ask questions to get intelligent, context-aware answers using both premium and free AI models.

## ✨ Features

- **Document Upload**: Support for PDF and text files
- **Intelligent Retrieval**: FAISS vector search with hybrid search options
- **Multiple AI Providers**: Choose between OpenAI (premium) and OpenRouter (free models)
- **Free Models Support**: Access Mistral, Zephyr, OpenChat, and other free models
- **Free Embeddings**: Automatic local embeddings when using OpenRouter-only
- **Conversational AI**: Context-aware responses with conversation memory
- **Modern UI**: Clean, responsive interface built with Streamlit
- **Source Citations**: View which documents informed each answer
- **Follow-up Questions**: AI-suggested questions to explore topics further
- **Export Functionality**: Download conversation history
- **Production Ready**: Containerized with Docker, modular architecture

## 🏗️ Architecture

```
rag-chatbot/
├── app.py                 # Streamlit frontend application
├── backend/
│   ├── embeddings.py      # Document processing & FAISS indexing
│   ├── retriever.py       # Document retrieval & similarity search
│   └── generator.py       # Answer generation with OpenAI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .env.example          # Environment variables template
└── README.md             # This documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- At least one API key:
  - OpenAI API key (for premium models)
  - OR OpenRouter API key (for free models) 
- Git (for cloning)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add at least one API key:
   OPENAI_API_KEY=your_openai_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Upload a document and start chatting!

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t rag-chatbot .
   ```

2. **Run the container**
   ```bash
   # With OpenAI API key
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_key rag-chatbot
   
   # With OpenRouter API key (for free models)
   docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_openrouter_key rag-chatbot
   
   # With both API keys
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_key -e OPENROUTER_API_KEY=your_openrouter_key rag-chatbot
   ```

3. **Access the application**
   - Open `http://localhost:8501` in your browser

### Docker Compose (Recommended)

1. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   services:
     rag-chatbot:
       build: .
       ports:
         - "8501:8501"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
       volumes:
         - ./data:/app/data  # Optional: persist FAISS indices
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional configurations
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
OPENROUTER_FREE_MODEL=mistralai/mistral-7b-instruct:free
```

### Getting API Keys

**OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login and navigate to API Keys
3. Create a new API key
4. Note: Requires payment for usage

**OpenRouter API Key (Free Models):**
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to API Keys and create one
4. Get free credits or use free models
5. No payment required for free tier models

### Supported Models

**OpenAI Models (Premium):**
- `gpt-3.5-turbo` (cost-effective, fast)
- `gpt-4` (higher quality, more expensive)
- `gpt-4-turbo-preview` (latest features)

**OpenRouter Free Models:**
- `mistralai/mistral-7b-instruct:free` (recommended free model)
- `huggingfaceh4/zephyr-7b-beta:free` (conversational)
- `openchat/openchat-7b:free` (general purpose)
- `gryphe/mythomist-7b:free` (creative writing)
- `nousresearch/nous-capybara-7b:free` (instruction following)

**Embedding Models:**

*OpenAI Embeddings (when OpenAI key provided):*
- `text-embedding-ada-002` (default, recommended)
- `text-embedding-3-small` (newer, smaller)
- `text-embedding-3-large` (newest, highest quality)

*Local Embeddings (automatic when OpenRouter-only):*
- `all-MiniLM-L6-v2` (sentence-transformers, free)
- Runs locally, no API calls for embeddings
- Good quality for most document types

## 💡 Usage Guide

### 1. Upload a Document
- Click "Choose a document" in the sidebar
- Select a PDF or text file (max recommended: 10MB)
- Click "Process Document" to create embeddings

### 2. Ask Questions
- Type your question in the chat input
- Get intelligent answers based on your document
- View sources that informed each answer

### 3. Explore Further
- Use suggested follow-up questions
- Adjust retrieval settings for different results
- Change AI model for different response styles

### 4. Advanced Features
- **Model Selection**: Choose between OpenAI premium and OpenRouter free models
- **Hybrid Search**: Combine semantic + keyword search
- **Export Chat**: Download conversation history
- **Multiple Documents**: Clear and upload new documents as needed

### 5. Cost Considerations

**Free Option (OpenRouter-only):**
- Free models available without payment
- Automatic local embeddings (no embedding costs)
- No OpenAI API key required
- Perfect for testing and personal use
- Decent quality responses for most use cases
- System automatically detects and uses free alternatives

**Premium Option (OpenAI):**
- Higher quality responses
- Faster response times
- Usage-based pricing
- Better for production applications

## 🔧 Advanced Configuration

### Retrieval Settings

**Number of Documents**: Adjust how many document chunks to retrieve (1-10)
- More chunks = more context but slower responses
- Recommended: 3-5 for most use cases

**Hybrid Search**: Enables combination of semantic and keyword search
- Better for specific terms and concepts
- Slightly slower but more comprehensive

### Generation Settings

**Temperature**: Controls response creativity (0.0-1.0)
- `0.0-0.2`: Focused, deterministic answers
- `0.3-0.7`: Balanced creativity and accuracy
- `0.8-1.0`: More creative but less predictable

## 🛠️ Development

### Project Structure

```
backend/
├── embeddings.py         # Document processing logic
│   ├── DocumentProcessor  # Handles file loading & chunking
│   └── create_vector_store # Creates FAISS indices
├── retriever.py          # Retrieval logic
│   ├── DocumentRetriever  # Basic similarity search
│   └── HybridRetriever   # Advanced hybrid search
└── generator.py          # Response generation
    ├── AnswerGenerator    # Basic response generation
    └── StreamingAnswerGenerator # Streaming responses
```

### Key Classes

**DocumentProcessor** (`backend/embeddings.py`)
- Loads PDF/text files
- Splits into chunks
- Creates embeddings with OpenAI
- Builds FAISS indices

**DocumentRetriever** (`backend/retriever.py`)
- Performs similarity search
- Retrieves relevant chunks
- Supports hybrid search

**AnswerGenerator** (`backend/generator.py`)
- Generates responses with OpenAI
- Maintains conversation context
- Suggests follow-up questions

### Adding Features

1. **New File Types**: Extend `DocumentProcessor.load_document()`
2. **Custom Retrievers**: Inherit from `DocumentRetriever`
3. **Different LLMs**: Modify `AnswerGenerator.__init__()`

## 🐛 Troubleshooting

### Common Issues

**"API Key not found"**
- Ensure `.env` file exists with `OPENAI_API_KEY`
- Check API key is valid and has credits
- Try entering API key directly in sidebar

**"Error processing document"**
- Check file format (PDF/TXT only)
- Ensure file isn't corrupted
- Try smaller file size

**"Slow responses"**
- Reduce number of retrieved documents
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Check internet connection

**Memory issues**
- Process smaller documents
- Reduce chunk overlap in `embeddings.py`
- Use Docker with memory limits

### Performance Optimization

1. **Reduce chunk size** in `DocumentProcessor`
2. **Lower embedding dimensions** (if using custom embeddings)
3. **Implement caching** for repeated queries
4. **Use streaming responses** for large answers

## 📝 API Reference

### DocumentProcessor

```python
processor = DocumentProcessor(openai_api_key, embedding_model)
documents = processor.process_uploaded_file(uploaded_file)
index, texts, docs = processor.create_embeddings(documents)
```

### DocumentRetriever

```python
retriever = DocumentRetriever(index, documents, embeddings)
results = retriever.retrieve_similar_documents(query, k=5)
context = retriever.get_context_for_query(query, k=5)
```

### AnswerGenerator

```python
generator = AnswerGenerator(openai_api_key, model_name, temperature)
answer = generator.generate_answer(query, context)
suggestions = generator.suggest_follow_up_questions(query, answer, context)
```

## 🔐 Security Considerations

- **API Keys**: Never commit API keys to version control
- **File Uploads**: Validate file types and sizes
- **Input Sanitization**: Sanitize user inputs
- **Rate Limiting**: Implement rate limiting for production
- **HTTPS**: Use HTTPS in production deployments

## 📊 Monitoring & Logging

For production deployments, consider:

- **Application Monitoring**: Use tools like New Relic or DataDog
- **Error Tracking**: Implement Sentry or similar
- **Usage Analytics**: Track document uploads and queries
- **Performance Metrics**: Monitor response times and costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG utilities
- [FAISS](https://faiss.ai/) for efficient vector search
- [OpenAI](https://openai.com/) for embeddings and chat models

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Consider sponsoring the project for priority support

---

**Happy chatting with your documents! 🎉**