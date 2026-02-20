import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv() 

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found! Please set it in your .env file.")
        self.model = None
        self.embedding_model=None

    def setModel(self, model_name="gemini-1.5-flash")->any:
        self.model_name = model_name    
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return self.model
    
    def setEmbeddingModel(self, model_name = "models/gemini-embedding-001")->any:
        self.embedding_model_name = model_name
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        return self.embedding_model
    
    def getModel(self) -> ChatGoogleGenerativeAI:
        if self.model is None:
            return self.setModel()
        return self.model
    
    def getEmbeddingModel(self) -> GoogleGenerativeAIEmbeddings:
        if self.embedding_model is None:
            return self.setEmbeddingModel()
        return self.embedding_model
    
