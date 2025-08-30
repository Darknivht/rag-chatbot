"""
Document retrieval module.
Handles similarity search and context retrieval from FAISS index.
"""

import numpy as np
import faiss
from typing import List, Tuple, Union
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class DocumentRetriever:
    """Handles document retrieval using FAISS similarity search."""
    
    def __init__(self, faiss_index, documents: List[Document], embeddings):
        """
        Initialize the document retriever.
        
        Args:
            faiss_index: FAISS index for similarity search
            documents: List of original Document objects
            embeddings: Embeddings instance (OpenAI or Local)
        """
        self.index = faiss_index
        self.documents = documents
        self.embeddings = embeddings
    
    def retrieve_similar_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k similar documents for a given query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        try:
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            # Retrieve corresponding documents
            retrieved_docs = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    # Convert distance to similarity score (higher is better)
                    similarity_score = 1 / (1 + distance)
                    retrieved_docs.append((self.documents[idx], similarity_score))
            
            return retrieved_docs
        
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    def get_context_for_query(self, query: str, k: int = 5, max_context_length: int = 3000) -> str:
        """
        Get concatenated context from retrieved documents for a query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            max_context_length: Maximum length of context to return
            
        Returns:
            Concatenated context string
        """
        try:
            retrieved_docs = self.retrieve_similar_documents(query, k)
            
            context_parts = []
            current_length = 0
            
            for doc, score in retrieved_docs:
                content = doc.page_content.strip()
                
                # Check if adding this content would exceed max length
                if current_length + len(content) > max_context_length:
                    # Add partial content if possible
                    remaining_length = max_context_length - current_length
                    if remaining_length > 100:  # Only add if there's meaningful space
                        content = content[:remaining_length] + "..."
                        context_parts.append(f"Context {len(context_parts) + 1}:\n{content}")
                    break
                
                context_parts.append(f"Context {len(context_parts) + 1}:\n{content}")
                current_length += len(content)
            
            return "\n\n".join(context_parts)
        
        except Exception as e:
            raise Exception(f"Error getting context: {str(e)}")
    
    def get_detailed_results(self, query: str, k: int = 5) -> List[dict]:
        """
        Get detailed retrieval results with metadata.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document details and scores
        """
        try:
            retrieved_docs = self.retrieve_similar_documents(query, k)
            
            results = []
            for i, (doc, score) in enumerate(retrieved_docs):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content.strip(),
                    'similarity_score': round(score, 4),
                    'metadata': doc.metadata,
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            raise Exception(f"Error getting detailed results: {str(e)}")


class HybridRetriever(DocumentRetriever):
    """Enhanced retriever with hybrid search capabilities."""
    
    def __init__(self, faiss_index, documents: List[Document], embeddings: OpenAIEmbeddings):
        super().__init__(faiss_index, documents, embeddings)
        # Create keyword index for hybrid search
        self._create_keyword_index()
    
    def _create_keyword_index(self):
        """Create simple keyword index for hybrid search."""
        self.keyword_index = {}
        for i, doc in enumerate(self.documents):
            words = doc.page_content.lower().split()
            for word in set(words):  # Use set to avoid duplicates
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append(i)
    
    def keyword_search(self, query: str, k: int = 10) -> List[int]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document indices
        """
        query_words = query.lower().split()
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_idx in self.keyword_index[word]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
        
        # Sort by score and return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_docs[:k]]
    
    def hybrid_retrieve(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            List of tuples containing (Document, combined_score)
        """
        # Get semantic search results
        semantic_results = self.retrieve_similar_documents(query, k * 2)
        semantic_scores = {i: score for i, (_, score) in enumerate(semantic_results)}
        
        # Get keyword search results
        keyword_indices = self.keyword_search(query, k * 2)
        keyword_scores = {idx: (len(keyword_indices) - rank) / len(keyword_indices) 
                         for rank, idx in enumerate(keyword_indices)}
        
        # Combine scores
        combined_scores = {}
        all_indices = set(range(min(len(semantic_results), len(self.documents))))
        
        for idx in all_indices:
            semantic_score = semantic_scores.get(idx, 0)
            keyword_score = keyword_scores.get(idx, 0)
            combined_score = (semantic_weight * semantic_score + 
                            (1 - semantic_weight) * keyword_score)
            combined_scores[idx] = combined_score
        
        # Sort and return top k results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for idx, score in sorted_results[:k]:
            if idx < len(self.documents):
                final_results.append((self.documents[idx], score))
        
        return final_results