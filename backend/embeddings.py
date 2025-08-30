"""
Document processing and embedding creation module.
Handles PDF/text file loading, chunking, and FAISS index creation.
"""

import os
import tempfile
import numpy as np
from typing import List, Optional

# Import with fallbacks for missing dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss-cpu not available. Please install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Please install with: pip install sentence-transformers")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain components not available. Please install with: pip install -r requirements.txt")


class LocalEmbeddings:
    """Local embeddings using sentence-transformers (free alternative to OpenAI)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings.
        
        Args:
            model_name: Sentence transformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for local embeddings. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()


class DocumentProcessor:
    """Handles document loading, processing, and embedding creation."""
    
    def __init__(self, openai_api_key: str = None, embedding_model: str = "text-embedding-ada-002", 
                 use_local_embeddings: bool = False):
        """
        Initialize the document processor.
        
        Args:
            openai_api_key: OpenAI API key for embeddings (optional if using local)
            embedding_model: OpenAI embedding model to use
            use_local_embeddings: Whether to use local embeddings instead of OpenAI
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain components are required. Install with: pip install -r requirements.txt")
        
        # Validate OpenAI API key if provided
        valid_openai_key = (openai_api_key and 
                           len(openai_api_key) > 20 and 
                           not openai_api_key.startswith("your_") and
                           openai_api_key != "your_openai_api_key_here")
        
        if use_local_embeddings or not valid_openai_key:
            self.embeddings = LocalEmbeddings()
            self.embedding_type = "local"
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model=embedding_model
            )
            self.embedding_type = "openai"
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load document based on file type.
        
        Args:
            file_path: Path to the document file
            file_type: Type of file ('pdf' or 'txt')
            
        Returns:
            List of Document objects
        """
        try:
            if file_type.lower() == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type.lower() in ['txt', 'text']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Process uploaded file from Streamlit.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of processed Document objects
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            file_type = uploaded_file.name.split('.')[-1]
            documents = self.load_document(tmp_file_path, file_type)
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def create_embeddings(self, documents: List[Document]) -> tuple:
        """
        Create embeddings for documents and build FAISS index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (faiss_index, document_texts, documents)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for vector search. Install with: pip install faiss-cpu")
        
        try:
            # Extract text from documents
            texts = [doc.page_content for doc in documents]
            
            # Create embeddings
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            return index, texts, documents
        
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def save_index(self, index, file_path: str):
        """
        Save FAISS index to file.
        
        Args:
            index: FAISS index object
            file_path: Path to save the index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for index operations. Install with: pip install faiss-cpu")
        
        try:
            faiss.write_index(index, file_path)
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, file_path: str):
        """
        Load FAISS index from file.
        
        Args:
            file_path: Path to the saved index
            
        Returns:
            FAISS index object
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for index operations. Install with: pip install faiss-cpu")
        
        try:
            return faiss.read_index(file_path)
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")


def create_vector_store(documents: List[Document], openai_api_key: str = None, 
                       use_local_embeddings: bool = False) -> tuple:
    """
    Convenience function to create vector store from documents.
    
    Args:
        documents: List of Document objects
        openai_api_key: OpenAI API key (optional if using local embeddings)
        use_local_embeddings: Whether to use local embeddings instead of OpenAI
        
    Returns:
        Tuple of (faiss_index, texts, documents, embeddings_instance)
    """
    # Validate OpenAI API key
    valid_openai_key = (openai_api_key and 
                       len(openai_api_key) > 20 and 
                       not openai_api_key.startswith("your_") and
                       openai_api_key != "your_openai_api_key_here")
    
    # Force local embeddings if no valid OpenAI key
    if not valid_openai_key:
        use_local_embeddings = True
    
    processor = DocumentProcessor(openai_api_key, use_local_embeddings=use_local_embeddings)
    index, texts, docs = processor.create_embeddings(documents)
    return index, texts, docs, processor.embeddings