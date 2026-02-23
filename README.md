## Overview

This is a Python-based **Self-RAG** system that implements adaptive retrieval-augmented generation with three reflection mechanisms to intelligently decide when to retrieve documents, grade their relevance, and verify answer quality.

## System Architecture

The system follows a 6-stage pipeline:

```
Query Input
    ↓
[STAGE 1] Retrieval Decision - Should we retrieve documents?
    ↓
[STAGE 2] Retrieval - Fetch candidate documents from vector store
    ↓
[STAGE 3] Relevance Grading - Score documents for relevance
    ↓
[STAGE 4] Context Assembly - Combine relevant documents
    ↓
[STAGE 5] Generation - Generate answer from context
    ↓
[STAGE 6] Support Checking - Verify answer is supported by context
    ↓
Final Answer with Confidence Metrics
```


## Installation

### Prerequisites
- Python 3.8+
- Google API Key for Gemini

### Step 1: Clone & Setup
```bash
cd "/Users/dev/Desktop/Self Rag"
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**step 3: install below libraries:**
# Core LangChain ecosystem
- pip install langchain
- pip install langchain-core
- pip install langchain-community
- pip install langchain-google-genai
- pip install langchain-chroma
- pip install langchain-text-splitters

# Vector database
pip install chromadb

# Document processing
pip install python-docx

# Environment variables
pip install python-dotenv

# Google API
pip install google-generativeai

# Additional utilities
pip install numpy
pip install requests

### Step 4: Configure API Key
Create a .env file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
---

## How to Run

### Run the Test Suite
```bash
python test.py
```


## Component Breakdown

### 1. **GeminiService.py** - LLM & Embedding Configuration
- Initializes Google's Gemini API connection
- Sets up the chat model (`gemini-3-pro-preview`)
- Configures embedding model (`gemini-embedding-001`)
- Manages API credentials from .env

### 2. **DocumentLoader.py** - Document Ingestion
- Loads `.docx` files using `Docx2txtLoader`
- Splits documents into chunks (100 chars with 20-char overlap)
- Returns list of `Document` objects ready for embedding

### 3. **chromaDb.py** - Vector Store Management
- Uses Chroma for vector similarity search
- Creates persistent vector store at `./vectorStore/chroma_langchain_db`
- Methods:
  - `addDocuments()` - Add docs and get IDs
  - `search()` - Retrieve top-k similar documents
  - `searchWithScore()` - Get documents with similarity scores
  - `deleteDocuments()` - Remove documents by ID

### 4. **reflectionCritics.py** - Three Reflection Mechanisms

#### Reflection 1: `shouldRetrieve()`
- Decides if retrieval is needed for the query
- Classification rules:
  - **NO retrieval**: Simple math, general knowledge, greetings
  - **YES retrieval**: Domain-specific facts, AI/ML/GenAI topics, citations needed
- Returns: `(should_retrieve: bool, confidence: float, reasoning: str)`

#### Reflection 2: `gradeRelevance()`
- Batches all retrieved documents and scores them 1-5:
  - **5** = Directly answers query
  - **4** = Highly relevant support
  - **3** = Somewhat relevant
  - **2** = Tangentially related
  - **1** = Not relevant
- Returns: `list[(Document, score, reasoning)]` (threshold = 3)

#### Reflection 3: `checkSupport()`
- Verifies if generated answer is supported by context
- Identifies unsupported/hallucinated claims
- Returns dict with:
  - `support_level`: FULLY_SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
  - `confidence`: 0.0-1.0
  - `is_supported`: Boolean flag

### 5. **selfrag.py** - Main Orchestration Engine
- **Configurable thresholds**:
  - `retrieval_confidence_threshold = 0.7`
  - `relevance_threshold = 3`
  - `top_k_retrieve = 7` documents
  - `max_context_length = 3000` chars

- **Key methods**:
  - `addDocuments()` - Load knowledge base
  - `query()` - Full pipeline with all reflections
  - `querySimple()` - Returns only answer string

### 6. **test.py** - Test Suite
- Loads sample GenAI document
- Tests 5 different queries to validate:
  - Smart retrieval decisions (math/general knowledge = no retrieve)
  - Domain queries (GenAI = retrieve)
  - Full reflection pipeline

---

## Data Flow Example

### Query: "What is GenAI?"

1. **Stage 1 - Retrieval Decision**
   - LLM analyzes: "GenAI-related → YES retrieve"
   - Confidence: 0.92

2. **Stage 2 - Retrieval**
   - Vector DB retrieves top 7 documents similar to query

3. **Stage 3 - Relevance Grading**
   - LLM scores: Doc1=5, Doc2=4, Doc3=2, Doc4=1, ...
   - Keeps scores ≥ 3 (3 docs pass)

4. **Stage 4 - Context Assembly**
   - Combines 3 relevant docs into 3000-char context

5. **Stage 5 - Generation**
   - LLM generates answer using only context

6. **Stage 6 - Support Checking**
   - LLM verifies each claim is in context
   - Returns: FULLY_SUPPORTED with 0.88 confidence

---

## File Structure
```
.
├── .env                          # API credentials
├── test.py                       # Test harness
├── modelconfig/
│   └── GeminiService.py         # LLM initialization
├── documentLoader/
│   └── DocumentLoader.py        # Document loading & chunking
├── vectorStore/
│   ├── chromaDb.py              # Vector DB operations
│   └── chroma_langchain_db/     # Persistent vector store
├── critics/
│   └── reflectionCritics.py     # Three reflection mechanisms
└── rag/
    └── selfrag.py               # Main orchestration
```

---

## Key Features

✅ **Smart Retrieval**: Skip retrieval for simple queries (math, general knowledge)  
✅ **Batch Relevance Grading**: Score all docs in one LLM call  
✅ **Hallucination Detection**: Verify answers are context-supported  
✅ **Confidence Metrics**: Get transparency on every decision  
✅ **Modular Design**: Each component is independent & reusable  