import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import os
import logging
import hashlib
import time
import json

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.collection_name = "document_embeddings"
        
        # Initialize components
        self._init_embedding_model()
        self._init_chromadb()
    
    def _init_embedding_model(self):
        """Initialize the embedding model with error handling"""
        try:
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise Exception(f"Failed to initialize embedding model: {e}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB with proper error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Using vector database directory: {self.persist_directory}")
            
            # Initialize ChromaDB client with better settings
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection with proper error handling
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"✅ Connected to existing collection: {self.collection_name}")
            except Exception:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise Exception(f"Failed to initialize vector database: {e}")
    
    def _generate_document_id(self, document: Document, index: int) -> str:
        """Generate a unique, consistent ID for a document"""
        # Create a hash from the document content and metadata
        content_hash = hashlib.md5(document.page_content.encode('utf-8')).hexdigest()[:8]
        filename = document.metadata.get('filename', 'unknown')
        
        # Clean filename for ID
        clean_filename = ''.join(c for c in filename if c.isalnum() or c in '-_')[:20]
        
        return f"{clean_filename}_{content_hash}_{index}"
    
    def _validate_document(self, document: Document) -> bool:
        """Validate document before adding to collection"""
        if not document.page_content or len(document.page_content.strip()) == 0:
            logger.warning("Skipping document with empty content")
            return False
        
        if len(document.page_content) > 50000:  # Limit very large documents
            logger.warning(f"Document too large ({len(document.page_content)} chars), truncating")
            document.page_content = document.page_content[:50000] + "..."
        
        return True
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata to ensure ChromaDB compatibility"""
        cleaned = {}
        for key, value in metadata.items():
            # Ensure string keys
            clean_key = str(key).replace('.', '_').replace(' ', '_')
            
            # Ensure values are simple types
            if isinstance(value, (str, int, float, bool)):
                cleaned[clean_key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                cleaned[clean_key] = ','.join(str(v) for v in value)
            else:
                # Convert complex objects to strings
                cleaned[clean_key] = str(value)
        
        return cleaned
    
    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """Add documents to the vector store with improved error handling"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        if not self.collection:
            logger.error("Collection not initialized")
            raise Exception("Vector store not properly initialized")
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Filter and validate documents
        valid_documents = []
        for doc in documents:
            if self._validate_document(doc):
                valid_documents.append(doc)
        
        if not valid_documents:
            logger.warning("No valid documents to add after filtering")
            return
        
        # Process documents in batches
        total_added = 0
        for i in range(0, len(valid_documents), batch_size):
            batch = valid_documents[i:i + batch_size]
            
            try:
                # Prepare batch data
                texts = []
                metadatas = []
                ids = []
                
                for j, doc in enumerate(batch):
                    # Generate unique ID
                    doc_id = self._generate_document_id(doc, total_added + j)
                    
                    # Check if document already exists
                    if self._document_exists(doc_id):
                        logger.debug(f"Document {doc_id} already exists, skipping")
                        continue
                    
                    texts.append(doc.page_content)
                    metadatas.append(self._clean_metadata(doc.metadata))
                    ids.append(doc_id)
                
                if not texts:
                    logger.info(f"Batch {i//batch_size + 1}: All documents already exist, skipping")
                    continue
                
                # Generate embeddings for the batch
                logger.info(f"Generating embeddings for batch {i//batch_size + 1} ({len(texts)} documents)...")
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
                
                # Add to collection
                logger.info(f"Adding batch {i//batch_size + 1} to collection...")
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(texts)
                logger.info(f"✅ Successfully added batch {i//batch_size + 1} ({len(texts)} documents)")
                
                # Small delay between batches to prevent overwhelming the system
                if i + batch_size < len(valid_documents):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Failed to add batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"✅ Successfully added {total_added} documents to vector store")
        
        if total_added == 0:
            logger.warning("⚠️ No new documents were added (all may already exist)")
    
    def _document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID already exists"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents with improved error handling"""
        if not self.collection:
            logger.error("Collection not initialized")
            return []
        
        try:
            # Check if collection has any documents
            count = self.collection.count()
            if count == 0:
                logger.info("No documents in collection for search")
                return []
            
            logger.info(f"Searching collection with {count} documents for query: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, count)  # Don't ask for more than available
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata or {},
                        'score': 1 - distance  # Convert distance to similarity score
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics with error handling"""
        try:
            if not self.collection:
                return {"total_documents": 0, "status": "not_initialized"}
            
            count = self.collection.count()
            
            # Get sample metadata to understand document types
            sample_result = self.collection.get(limit=min(10, count))
            
            file_types = {}
            if sample_result['metadatas']:
                for metadata in sample_result['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                "total_documents": count,
                "status": "active",
                "file_types": file_types,
                "collection_name": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"total_documents": 0, "status": "error", "error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            if not self.collection:
                logger.warning("Collection not initialized")
                return False
            
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                # Delete all documents
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"✅ Cleared {len(all_docs['ids'])} documents from collection")
            else:
                logger.info("Collection was already empty")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete and recreate)"""
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False
            
            # Delete existing collection
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.info(f"Collection didn't exist or couldn't be deleted: {e}")
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reset collection: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the vector store"""
        health = {
            "status": "unknown",
            "embedding_model": False,
            "chromadb_client": False,
            "collection": False,
            "can_search": False,
            "document_count": 0
        }
        
        try:
            # Check embedding model
            if self.embedding_model:
                test_embedding = self.embedding_model.encode(["test"])
                health["embedding_model"] = len(test_embedding) > 0
            
            # Check ChromaDB client
            if self.client:
                health["chromadb_client"] = True
            
            # Check collection
            if self.collection:
                health["collection"] = True
                health["document_count"] = self.collection.count()
            
            # Test search capability
            if health["embedding_model"] and health["collection"]:
                try:
                    self.similarity_search("test query", k=1)
                    health["can_search"] = True
                except Exception:
                    pass
            
            # Overall status
            if all([health["embedding_model"], health["chromadb_client"], 
                   health["collection"], health["can_search"]]):
                health["status"] = "healthy"
            elif health["embedding_model"] and health["chromadb_client"]:
                health["status"] = "partial"
            else:
                health["status"] = "unhealthy"
                
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
        
        return health