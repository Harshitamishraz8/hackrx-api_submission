import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from .utils import download_pdf, extract_text_from_pdf, chunk_text

load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
)

# Initialize embedding model
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
index_name = os.getenv("PINECONE_INDEX", "hackrx-index")

def get_pinecone_index():
    """Get or create Pinecone index"""
    try:
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            print(f"Creating Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine"
            )
        
        return pinecone.Index(index_name)
    except Exception as e:
        print(f"Error with Pinecone index: {e}")
        raise

async def process_and_store_pdf(pdf_url: str, doc_id: str) -> bool:
    """Process PDF and store embeddings in Pinecone"""
    try:
        index = get_pinecone_index()
        
        # Check if document already exists
        try:
            existing = index.fetch(ids=[f"{doc_id}-0"])
            if existing and existing.vectors:
                print(f"Document {doc_id} already exists in Pinecone")
                return True
        except:
            pass  # Document doesn't exist, continue processing
        
        # Download and extract text from PDF
        pdf_bytes = download_pdf(pdf_url)
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text or len(text.strip()) < 100:
            print("Insufficient text extracted from PDF")
            return False
        
        # Split into chunks
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        if not chunks:
            print("No valid chunks created from PDF")
            return False
        
        # Generate embeddings and store in Pinecone
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = embedding_model.encode(chunk).tolist()
                
                # Prepare vector for Pinecone
                vector_id = f"{doc_id}-{i}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "url": pdf_url
                    }
                })
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        if not vectors_to_upsert:
            print("No vectors to upsert")
            return False
        
        # Upsert vectors to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
        
        print(f"Successfully stored {len(vectors_to_upsert)} vectors for document {doc_id}")
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False