import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv() 

class geminiService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found! Please set it in your .env file.")

    def setModel(self, model_name)->any:
        self.model_name = model_name    
        model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            seed=42
        )
        return model
    
    def setEmbeddingModel(self, model_name = "models/gemini-embedding-001")->any:
        self.embedding_model_name = model_name
        embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        return embeddings
    
    def createAgent(self, tools, model)->any:
        agent = create_agent(model=model, tools=tools)
        return agent