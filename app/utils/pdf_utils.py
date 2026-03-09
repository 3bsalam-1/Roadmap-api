"""PDF utility functions for the roadmap API."""
import os
from pathlib import Path
from typing import List


def list_pdfs(directory: str) -> List[str]:
    """
    List all PDF files in a directory.
    
    Args:
        directory: Path to the directory containing PDFs
        
    Returns:
        List of absolute paths to PDF files
    """
    pdf_dir = Path(directory)
    
    if not pdf_dir.exists():
        return []
    
    return [str(pdf) for pdf in pdf_dir.glob("*.pdf")]


def validate_pdf(filename: str) -> bool:
    """
    Validate that a PDF file exists and is readable.
    
    Args:
        filename: Path to the PDF file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(filename):
        return False
    
    if not filename.lower().endswith(".pdf"):
        return False
    
    return True


def get_track_from_filename(filename: str) -> str:
    """
    Extract track name from PDF filename.
    
    Args:
        filename: PDF filename (e.g., "machine_learning.pdf")
        
    Returns:
        Track name (e.g., "machine learning")
    """
    # Remove .pdf extension and replace underscores with spaces
    name = os.path.basename(filename).replace(".pdf", "").replace("_", " ")
    return name.lower().strip()
