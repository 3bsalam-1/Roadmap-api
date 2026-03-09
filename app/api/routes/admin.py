import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from app.core.security import require_admin
from app.core.config import settings
from app.core.logging import logger
from app.utils.cloudinary_utils import upload_pdf_to_cloudinary, list_cloudinary_pdfs
from app.rag import ingestor
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/pdfs/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    track_name: str = Form(...),
    _: None = Depends(require_admin),
):
    """
    Upload a PDF to Cloudinary, then ingest into Pinecone.
    
    Protected by API key authentication.
    
    Args:
        file: PDF file (multipart/form-data)
        track_name: Name of the track (e.g., "machine learning")
        
    Returns:
        JSON with track, filename, cloudinary_url, and chunks_indexed count
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # Read file bytes
    file_bytes = await file.read()
    
    # Upload to Cloudinary
    try:
        cloudinary_url = upload_pdf_to_cloudinary(file_bytes, file.filename)
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {e}")
        raise HTTPException(status_code=502, detail=f"Cloudinary upload failed: {str(e)}")
    
    # Write to temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        # Load and chunk the PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(pages)
        
        # Ingest into Pinecone
        count = ingestor.ingest_documents(chunks, track_name, file.filename)
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        # Return the Cloudinary URL so the file is not lost
        return {
            "track": track_name,
            "filename": file.filename,
            "cloudinary_url": cloudinary_url,
            "chunks_indexed": 0,
            "error": f"PDF processing failed: {str(e)}",
        }
    finally:
        os.unlink(tmp_path)
    
    return {
        "track": track_name,
        "filename": file.filename,
        "cloudinary_url": cloudinary_url,
        "chunks_indexed": count,
    }


@router.get("/pdfs")
def list_pdfs(_: None = Depends(require_admin)):
    """
    List all PDFs stored in Cloudinary.
    
    Protected by API key authentication.
    """
    return {"pdfs": list_cloudinary_pdfs()}
