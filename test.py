from rag.selfrag import SelfRag
from langchain_core.documents import Document
from documentLoader.DocumentLoader import DocumentLoader

def main():
    # Initialize Self-RAG system
    rag = SelfRag()
    
    doc=DocumentLoader("/Users/dev/Downloads/Generative AI DevSinc.docx")
    documents = doc.load_and_split()
    
    add_result = rag.addDocuments(documents)
    print(f"Added {len(add_result['document_ids'])} documents")
    print()
    
    # Test queries
    test_queries = [
        "What is 2 + 2?",  # Should NOT retrieve (simple math)
        "Who won the World Cup in 2022?",  # Should NOT retrieve (general knowledge)
        "what is the Outline & Learning Path for gen ai",  # SHOULD retrieve
        "What is GenAI?",  # Should retrieve
        "What aree large language models?",  # Should retrieve and find relevant docs
    ]
    
    for query in test_queries:
        print(f"{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        result = rag.query(query)
        
        print(f"\nANSWER:\n{result['answer']}\n")
        
        print("REFLECTION METADATA:")
        print(f"  Retrieval used: {result['reflection'].get('retrieval_used', 'N/A')}")
        
        if 'retrieval_decision' in result['reflection']:
            rd = result['reflection']['retrieval_decision']
            print(f"  Retrieval decision: {'RETRIEVE' if rd['should_retrieve'] else 'SKIP'} (confidence: {rd['confidence']:.2f})")
        
        if 'relevance_grading' in result['reflection']:
            rg = result['reflection']['relevance_grading']
            print(f"  Relevant docs: {rg['relevant_count']}/{rg['total_retrieved']}")
        
        if 'support_check' in result['reflection'] and 'support_level' in result['reflection']['support_check']:
            sc = result['reflection']['support_check']
            print(f"  Support level: {sc['support_level']} (confidence: {sc.get('confidence', 0):.2f})")
        
        print()

if __name__ == "__main__":
    main()