from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import re

class ReflectionCritics:
    """
    Implements the three Self-RAG reflection mechanisms:
    1. Retrieval Decision (should we retrieve?)
    2. Relevance Grading (are retrieved docs relevant?)
    3. Support Checking (is answer supported by context?)
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    def shouldRetrieve(self, query: str, confidence_threshold: float = 0.7) -> tuple[bool, float, str]:
        """
        REFLECTION 1: Decide if retrieval is necessary.
        """
        prompt = f"""You are a retrieval decision system. Analyze if this query requires external document retrieval.
            Query: "{query}"
            Classification rules:
            - Answer "NO" if: simple math, general knowledge, greetings, personal opinions
            - Answer "YES" if: requires specific facts, recent events, domain knowledge, citations
            Respond in this exact format:
            DECISION: YES or NO
            CONFIDENCE: 0.0 to 1.0
            REASONING: Brief explanation
            Your response:"""
        
        try:
            response = self.llm.invoke(prompt).content
            
            decision_match = re.search(r'DECISION:\s*(YES|NO)', response, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            decision = decision_match.group(1).upper() if decision_match else "YES"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            should_retrieve = decision == "YES" or confidence < confidence_threshold
            
            return should_retrieve, confidence, reasoning
            
        except Exception as e:
            return True, 0.0, f"Error in retrieval decision: {str(e)}"
    
    def gradeRelevance(self, query: str, documents: list[Document], threshold: int = 3) -> list[tuple[Document, int, str]]:
        """
        REFLECTION 2: Grade each retrieved document's relevance.
        """
        relevant_docs = []
        
        for doc in documents:
            content_preview = doc.page_content[:800]
            
            prompt = f"""Grade the relevance of this document to the query.
                Query: "{query}"

                Document excerpt:
                {content_preview}

                Rate on scale 1-5:
                5 = Directly answers the query
                4 = Highly relevant supporting info
                3 = Somewhat relevant
                2 = Tangentially related
                1 = Not relevant

                Respond in this exact format:
                SCORE: [1-5]
                REASONING: Brief explanation

                Your response:
            """
            
            try:
                response = self.llm.invoke(prompt).content
                
                score_match = re.search(r'SCORE:\s*([1-5])', response)
                reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
                
                score = int(score_match.group(1)) if score_match else 1
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
                
                if score >= threshold:
                    relevant_docs.append((doc, score, reasoning))
                    
            except Exception as e:
                relevant_docs.append((doc, threshold, f"Error grading: {str(e)}"))
        
        # Sort by score descending
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return relevant_docs
    
    def checkSupport(self, query: str, answer: str, context: str) -> dict:
        """
        REFLECTION 3: Verify if generated answer is supported by retrieved context.
        """
        prompt = f"""You are a fact-checking system. Verify if the answer is fully supported by the context.
            Context:
            {context[:2000]}

            Question: "{query}"

            Answer:
            {answer}

            Analysis tasks:
            1. Identify all factual claims in the answer
            2. Check if EACH claim is supported by the context
            3. List any unsupported or hallucinated claims

            Respond in this exact format:
            SUPPORT_LEVEL: FULLY_SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED
            CONFIDENCE: 0.0 to 1.0
            UNSUPPORTED_CLAIMS: List each unsupported claim (or "None")
            REASONING: Brief explanation

            Your response:
        """
        
        try:
            response = self.llm.invoke(prompt).content
            
            support_match = re.search(r'SUPPORT_LEVEL:\s*(FULLY_SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED)', response, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
            claims_match = re.search(r'UNSUPPORTED_CLAIMS:\s*(.+?)(?=REASONING:|$)', response, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            support_level = support_match.group(1).upper() if support_match else "UNSUPPORTED"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            unsupported_claims = claims_match.group(1).strip() if claims_match else "Unknown"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
            
            return {
                "support_level": support_level,
                "confidence": confidence,
                "unsupported_claims": unsupported_claims,
                "reasoning": reasoning,
                "is_supported": support_level == "FULLY_SUPPORTED" and confidence >= 0.7
            }
            
        except Exception as e:
            return {
                "support_level": "ERROR",
                "confidence": 0.0,
                "unsupported_claims": f"Error: {str(e)}",
                "reasoning": "Support check failed",
                "is_supported": False
            }