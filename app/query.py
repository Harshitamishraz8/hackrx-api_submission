import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
)

# Initialize embedding model
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
index_name = os.getenv("PINECONE_INDEX", "hackrx-index")

def query_documents(question: str, doc_id: str, top_k: int = 5) -> list:
    """Query Pinecone for relevant document chunks"""
    try:
        index = pinecone.Index(index_name)
        
        # Generate embedding for the question
        question_embedding = embedding_model.encode(question).tolist()
        
        # Query Pinecone with document filter
        query_response = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        # Extract text from results
        contexts = []
        for match in query_response.matches:
            if match.score > 0.3:  # Only include relevant matches
                text = match.metadata.get("text", "")
                if text and text not in contexts:
                    contexts.append(text)
                    print(f"Found relevant context (score: {match.score:.3f}): {text[:100]}...")
        
        if not contexts:
            print(f"No relevant contexts found for question: {question}")
            # Fallback: get any chunks from this document
            fallback_response = index.query(
                vector=question_embedding,
                top_k=3,
                include_metadata=True,
                filter={"doc_id": {"$eq": doc_id}}
            )
            
            for match in fallback_response.matches:
                text = match.metadata.get("text", "")
                if text:
                    contexts.append(text)
                    print(f"Using fallback context: {text[:100]}...")
        
        return contexts
        
    except Exception as e:
        print(f"Error querying documents: {e}")
        return []