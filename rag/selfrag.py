from modelconfig.GeminiService import GeminiService
from vectorStore.chromaDb import ChromaDb
from critics.reflectionCritics import ReflectionCritics
from langchain_core.documents import Document

class SelfRag:
    """
    Self-Reflective Retrieval-Augmented Generation System.
    
    Implements adaptive retrieval with three reflection stages:
    1. Pre-retrieval: Decide if retrieval is needed
    2. Post-retrieval: Grade document relevance
    3. Post-generation: Verify answer support
    """
    
    def __init__(self,model_name: str = "gemini-1.5-flash",temperature: float = 0.1,collection_name: str = "test_collection"):
        self.gemini_service = GeminiService()
        self.llm = self.gemini_service.setModel(model_name)
        self.chroma_db = ChromaDb()
        self.vector_store = self.chroma_db.createVectorStore()
        self.critics = ReflectionCritics(self.llm)
        self.retrieval_confidence_threshold = 0.7
        self.relevance_threshold = 3
        self.top_k_retrieve = 5
        self.max_context_length = 3000
    
    def addDocuments(self, documents: list[Document]) -> dict:
        """
        Add documents to knowledge base.
        """
        try:
            ids = self.chroma_db.addDocuments(documents)
            return {
                "success": True,
                "document_ids": ids,
                "total_documents": self.chroma_db.getDocumentCount()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def query(self, question: str, force_retrieve: bool = False) -> dict:
        """
        Main Self-RAG query pipeline with all reflection stages.
        """
        result = {
            "question": question,
            "answer": None,
            "reflection": {},
            "debug": {}
        }
        
        # ===== STAGE 1: RETRIEVAL DECISION =====
        if not force_retrieve:
            should_retrieve, confidence, reasoning = self.critics.shouldRetrieve(
                question, 
                self.retrieval_confidence_threshold
            )
            
            result["reflection"]["retrieval_decision"] = {
                "should_retrieve": should_retrieve,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
            # If confident we don't need retrieval, answer directly
            if not should_retrieve and confidence >= self.retrieval_confidence_threshold:
                direct_answer = self.llm.invoke(question).content
                result["answer"] = direct_answer
                result["reflection"]["retrieval_used"] = False
                result["reflection"]["support_check"] = {
                    "note": "No retrieval performed - answered from parametric knowledge"
                }
                return result
        
        # ===== STAGE 2: RETRIEVAL =====
        retrieved_docs = self.chroma_db.search(question, k=self.top_k_retrieve)
        result["debug"]["retrieved_count"] = len(retrieved_docs)
        
        if not retrieved_docs:
            result["answer"] = "I apologize, but I couldn't find relevant information in my knowledge base to answer this question."
            result["reflection"]["retrieval_used"] = True
            result["reflection"]["relevant_docs_found"] = 0
            return result
        
        # ===== STAGE 3: RELEVANCE GRADING =====
        relevant_docs_scored = self.critics.gradeRelevance(
            question, 
            retrieved_docs, 
            self.relevance_threshold
        )
        
        result["reflection"]["relevance_grading"] = {
            "total_retrieved": len(retrieved_docs),
            "relevant_count": len(relevant_docs_scored),
            "scores": [(score, reasoning) for _, score, reasoning in relevant_docs_scored]
        }
        
        if not relevant_docs_scored:
            result["answer"] = "I found documents but none were sufficiently relevant to answer your question confidently."
            result["reflection"]["retrieval_used"] = True
            result["reflection"]["relevant_docs_found"] = 0
            return result
        
        # ===== STAGE 4: CONTEXT ASSEMBLY =====
        # Use only relevant documents, truncate if needed
        context_parts = []
        total_length = 0
        
        for doc, score, _ in relevant_docs_scored:
            content = doc.page_content
            if total_length + len(content) > self.max_context_length:
                # Truncate last document if needed
                remaining = self.max_context_length - total_length
                content = content[:remaining]
                context_parts.append(content)
                break
            context_parts.append(content)
            total_length += len(content)
        
        context = "\n\n---\n\n".join(context_parts)
        result["debug"]["context_length"] = len(context)
        
        # ===== STAGE 5: GENERATION =====
        generation_prompt = f"""You are a helpful assistant. Answer the question using ONLY the information provided in the context below.s
            Context:
            {context}

            Question: {question}

            Instructions:
            - Answer based only on the context above
            - If the context doesn't contain enough information, explicitly state this
            - Be concise and factual
            - Do not add information not present in the context

            Answer:"""
        
        answer = self.llm.invoke(generation_prompt).content
        result["answer"] = answer
        result["reflection"]["retrieval_used"] = True
        result["reflection"]["relevant_docs_found"] = len(relevant_docs_scored)
        
        # ===== STAGE 6: SUPPORT CHECKING =====
        support_check = self.critics.checkSupport(question, answer, context)
        result["reflection"]["support_check"] = support_check
        
        # Add warning if answer not fully supported
        if not support_check["is_supported"]:
            result["answer"] = f"{answer}\n\n⚠️ WARNING: This answer may contain unsupported claims. Confidence: {support_check['confidence']:.2f}"
        
        return result
    
    def querySimple(self, question: str) -> str:
        """
        Simplified query method that returns only the answer string.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        result = self.query(question)
        return result["answer"]