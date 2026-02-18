import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from modelconfig.GeminiService import geminiService
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv() 

class chromaDb(self):
    count=0
    def __init__(self):
        self.gemini_service = geminiService()
        self.embedding_model = self.gemini_service.setEmbeddingModel()

    def createVectorStore(self)->any:
        if self.vector_store is not None:
            return self.vector_store
        else:     
            self.vector_store = Chroma(
                collection_name="test_collection",
                embedding_function=self.embedding_model,
                persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
            )
            return self.vector_store

    def addDocuments(self, documents):
        self.vector_store.add_documents(documents=documents,ids=f"id{self.count}")
        self.count += 1

    def deleteDocuments(self, ids):
        self.vector_store.delete(ids=ids)
         

