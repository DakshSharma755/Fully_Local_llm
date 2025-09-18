import os
import lancedb
import llm_core
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers.cross_encoder import CrossEncoder


load_dotenv()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 
LANCEDB_DIR = "./data/lancedb" 
TABLE_NAME = "rag_documents"
SOURCE_DOCS_DIR = "./data"

model = None
db = None
table = None
cross_encoder = None
text_splitter = None

def initialize():
    """
    Initializes the embedding model, the Gemini AI model, and the LanceDB vector store.
    If the vector store is empty, it populates it by processing documents
    from the source directory.
    """
    global model, db, table, cross_encoder,text_splitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150,
        length_function=len,
    )

    print("🚀 Initializing RAG Core...")

    print(f"   - Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')

    
    llm_core.initialize_llm()
    
    print("   - Loading Cross-Encoder model for re-ranking...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

    print(f"   - Connecting to LanceDB at: {LANCEDB_DIR}...")
    db = lancedb.connect(LANCEDB_DIR)

    # Check if the table already exists
    table_names = db.table_names()
    if TABLE_NAME in table_names:
        print(f"   - Found existing table '{TABLE_NAME}'. Loading it.")
        table = db.open_table(TABLE_NAME)
    else:
        print(f"   - Table '{TABLE_NAME}' not found. Indexing documents...")
        table = create_and_index_documents()

    print("✅ RAG Core Initialized Successfully!")


def create_and_index_documents():
    """
    Scans the source documents directory for PDFs, splits them into chunks,
    generates embeddings, and creates a new LanceDB table with the data.
    """

    documents = []
    source_path = Path(SOURCE_DOCS_DIR)
    
    print(f"   - Scanning for PDF documents in '{SOURCE_DOCS_DIR}'...")
    
    for file_path in source_path.rglob("*.pdf"):
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            chunks = text_splitter.split_text(full_text)
            
            # Create a document for each chunk
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "source": f"{file_path.name} (chunk {i+1})"
                })
            
            print(f"      - Processed and chunked {file_path.name} into {len(chunks)} chunks.")
        except Exception as e:
            print(f"      - Error processing {file_path}: {e}")

    if not documents:
        print("   - No documents found. The application will exit as there is no data to index.")
        exit()

    print(f"   - Found {len(documents)} total chunks. Generating embeddings...")
    
    all_texts = [doc['text'] for doc in documents]
    embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
    
    for i, doc in enumerate(documents):
        doc['vector'] = embeddings[i]

    print("   - Creating new LanceDB table and adding data...")
    new_table = db.create_table(TABLE_NAME, data=documents, mode="overwrite")
    
    return new_table

def get_answer(query: str):
    """
    Takes a user query, finds relevant documents, re-ranks them for better relevance,
    and uses Gemini to generate a final answer.
    """
    if not all([model, table, cross_encoder]):
        raise RuntimeError("RAG Core is not initialized. Call initialize() first.")

    query_vector = model.encode(query)
    try:
        initial_results = table.search(query_vector).limit(10).to_list()
    except Exception as e:
        print(f"Error during initial LanceDB search: {e}")
        initial_results = []

    if not initial_results:
        context = "No relevant documents found."
    else:
        print(f"   - Re-ranking {len(initial_results)} initial results...")
        
        pairs = [[query, doc['text']] for doc in initial_results]
        
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        for i in range(len(initial_results)):
            initial_results[i]['rerank_score'] = scores[i]
            
        reranked_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        top_k = 3
        final_context_docs = reranked_results[:top_k]
        context = "\n---\n".join([doc['text'] for doc in final_context_docs])
        print(f"   - Top {top_k} sources after re-ranking: {[doc['source'] for doc in final_context_docs]}")


    prompt = f"""
        <|user|>
        You are a helpful and creative AI assistant. Use the provided context to inform your answer. Synthesize the information from the documents with your own general knowledge and reasoning abilities to provide a comprehensive response.

        CONTEXT:
        {context}

        QUESTION:
        {query}
        <|end|>
        <|assistant|>
        """
    return llm_core.generate_answer(prompt)
    
def process_and_add_document(file_path: Path):
    """
    Processes a single PDF file, chunks it, creates embeddings,
    and adds the data to the existing LanceDB table.
    """
    global table, model, text_splitter 

    if not all([table, model, text_splitter]):
        print("   - Core components not initialized. Cannot add document.")
        return False
        
    try:
        print(f"   - Processing new document: {file_path.name}")
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        chunks = text_splitter.split_text(full_text)
        
        if not chunks:
            print(f"   - No text found in {file_path.name}. Skipping.")
            return False

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": f"{file_path.name} (chunk {i+1})"
            })
        
        print(f"      - Chunked into {len(chunks)} parts. Generating embeddings...")
        all_texts = [doc['text'] for doc in documents]
        embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
        
        for i, doc in enumerate(documents):
            doc['vector'] = embeddings[i]

        table.add(documents)
        print(f"      - Successfully added {file_path.name} to the vector store.")
        return True
    except Exception as e:
        print(f"      - Error processing and adding document {file_path.name}: {e}")
        return False