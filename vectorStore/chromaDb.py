import os
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from modelconfig.GeminiService import GeminiService
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv() 

class ChromaDb:
    def __init__(self):
        self.gemini_service = GeminiService()
        self.embedding_model = self.gemini_service.setEmbeddingModel()
        self.vector_store = None
        self.count = 0

    def createVectorStore(self)->any:
        if not hasattr(self, 'vector_store') or self.vector_store is None:   
            self.vector_store = Chroma(
                collection_name="test_collection",
                embedding_function=self.embedding_model,
                persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
            )
        return self.vector_store

    def addDocuments(self, documents: list) -> list[str]:
            """Add documents and return their IDs"""
            if self.vector_store is None:
                raise ValueError("Vector store not initialized. Call createVectorStore() first.")
            
            ids = [f"doc_{self.count + i}" for i in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=ids)
            self.count += len(documents)
            return ids
        
    def deleteDocuments(self, ids: list[str]):
            """Delete documents by IDs"""
            if self.vector_store is None:
                raise ValueError("Vector store not initialized.")
            self.vector_store.delete(ids=ids)
        
    def search(self, query: str, k: int = 5) -> list:
            """Retrieve top-k documents for a query"""
            if self.vector_store is None:
                raise ValueError("Vector store not initialized.")
            return self.vector_store.similarity_search(query, k=k)
    
    def searchWithScore(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call createVectorStore() first.")
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def getDocumentCount(self) -> int:
        """Return total number of documents added"""
        return self.count
         

