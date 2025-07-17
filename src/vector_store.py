import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain.schema import Document
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "document_embeddings"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            print("No documents to add")
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Create unique IDs
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search similar documents
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': 1 - distance  # Convert distance to similarity score
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {"total_documents": count}