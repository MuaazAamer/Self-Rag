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
        prompt = f"""You are an expert retrieval decision system. Analyze if this query requires external document retrieval.
            Query: "{query}"
            Classification rules:
            - Answer "NO" if: simple math, general knowledge, greetings, personal opinions,anything unrelated to AI,ML,GEN AI.
            - Answer "YES" if: requires specific facts, recent events, domain knowledge, citations,links and anything related to AI,ML,GEN AI.
            Respond in this exact format,do not generate any additional text:
            DECISION: YES or NO
            CONFIDENCE: 0.0 to 1.0
            REASONING: Brief explanation
            """
        
        try:
            response = self.llm.invoke(prompt).content
            response=response[0]["text"]
            
            decision_match = re.search(r'DECISION:\s*(YES|NO)', response, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            decision = decision_match.group(1).upper() if decision_match else "YES"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            should_retrieve = decision == "YES"
            
            return should_retrieve, confidence, reasoning
            
        except Exception as e:
            return True, 0.0, f"Error in retrieval decision: {str(e)}"
    
    def gradeRelevance(
        self,
        query: str,
        documents: list[Document],
        threshold: int = 3
    ) -> list[tuple[Document, int, str]]:
        """
        REFLECTION 2 (BATCHED): Grade all retrieved documents in one LLM call.
        Returns same format as before:
            list[(Document, score, reasoning)]
        """

        if not documents:
            return []

        previews = []
        for i, doc in enumerate(documents):
            preview = doc.page_content[:800]
            previews.append(f"DOCUMENT_{i}:\n{preview}")

        prompt = f"""
            You are a relevance grading system.

            Query: "{query}"

            Below are numbered document excerpts.

            {chr(10).join(previews)}

            Rate each document from 1-5:

            5 = Directly answers the query  
            4 = Highly relevant supporting info  
            3 = Somewhat relevant  
            2 = Tangentially related  
            1 = Not relevant  

            Respond ONLY with valid JSON in this exact format:

            [
            {{"doc_id": 0, "score": 1-5, "reasoning": "..."}},
            ...
            ]
        """

        try:
            response = self.llm.invoke(prompt).content
            if isinstance(response, list):
                response = response[0].get("text", "")

            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")

            import json
            results = json.loads(json_match.group(0))

            relevant_docs = []

            for item in results:
                doc_id = item.get("doc_id")
                score = int(item.get("score", 0))
                reasoning = item.get("reasoning", "No reasoning")

                if (
                    isinstance(doc_id, int)
                    and 0 <= doc_id < len(documents)
                    and score >= threshold
                ):
                    relevant_docs.append(
                        (documents[doc_id], score, reasoning)
                    )

            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            return relevant_docs

        except Exception as e:
            return [
                (doc, threshold, f"Batch grading error: {str(e)}")
                for doc in documents
            ]
    
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
            response=response[0]["text"]
            
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