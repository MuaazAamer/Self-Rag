from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_and_split(self) -> list:
        loader = Docx2txtLoader(self.file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents(data)
        
        return texts