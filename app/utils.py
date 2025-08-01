import requests
import hashlib
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import re

def download_pdf(url: str) -> bytes:
    """Download PDF from URL and return bytes"""
    try:
        print(f"Downloading PDF from: {url}")
        
        # Handle Google Drive URLs
        if "drive.google.com" in url and "uc?export=download" not in url:
            # Extract file ID from various Google Drive URL formats
            file_id = None
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            
            if file_id:
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        if not response.content.startswith(b'%PDF'):
            raise ValueError("Downloaded content is not a valid PDF")
        
        print(f"Successfully downloaded PDF ({len(response.content)} bytes)")
        return response.content
        
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        raise

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfminer"""
    try:
        print("Extracting text from PDF...")
        
        # Use pdfminer for better text extraction
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        )
        
        text = extract_text(BytesIO(pdf_bytes), laparams=laparams)
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
        text = text.strip()
        
        print(f"Extracted {len(text)} characters from PDF")
        print(f"First 500 characters: {text[:500]}")
        
        if len(text) < 100:
            raise ValueError("Extracted text is too short, PDF might be image-based or corrupted")
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks"""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = -1
            for i in range(max(0, end - 100), end):
                if text[i] in '.!?':
                    sentence_end = i + 1
                    break
            
            if sentence_end > start:
                end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:  # Only add substantial chunks
            chunks.append(chunk)
        
        start = end - overlap
        
        if start >= len(text):
            break
    
    print(f"Created {len(chunks)} text chunks")
    return chunks

def generate_doc_id(url: str) -> str:
    """Generate a unique document ID from URL"""
    return hashlib.md5(url.encode()).hexdigest()[:8]