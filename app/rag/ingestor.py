import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from app.core.config import settings
from app.core.logging import logger
from app.rag.retriever import get_embeddings, get_index


# Track name aliases for normalization
TRACK_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "data sci": "data science",
    "web dev": "web development",
    "devops": "devops",
    "cloud": "cloud computing",
    "mobile dev": "mobile development",
    "frontend": "frontend development",
    "backend": "backend development",
    "full stack": "full stack development",
}


def normalize_track_name(track_query: str) -> str:
    """Normalize track name by lowercasing and handling aliases."""
    normalized = track_query.lower().strip()
    # Check for aliases
    for alias, full_name in TRACK_ALIASES.items():
        if alias in normalized or normalized == alias:
            return full_name
    return normalized


def extract_track_from_filename(filename: str) -> str:
    """Extract track name from PDF filename (e.g., machine_learning.pdf -> machine learning)."""
    # Remove .pdf extension and replace underscores with spaces
    name = filename.replace(".pdf", "").replace("_", " ")
    return name.lower().strip()


def check_exists_in_pinecone(source_filename: str) -> bool:
    """Check if a PDF source already exists in Pinecone using metadata filter."""
    try:
        index = get_index()
        # Query with a dummy vector but filter by source
        results = index.query(
            vector=[0.0] * 384,  # Dummy vector
            top_k=1,
            filter={"source": source_filename},
            include_metadata=True,
        )
        if results.get("matches"):
            logger.info(f"[SKIP] {source_filename} already in Pinecone.")
            return True
        return False
    except Exception as e:
        logger.warning(f"Could not check existing documents: {e}")
        return False


def ingest_documents(docs: List[Document], track: str, source_filename: str) -> int:
    """
    Chunk and embed documents, upsert into Pinecone.
    Idempotency: use Pinecone metadata filter to check if source_filename
    already exists before upserting. Skip if found.
    """
    # Check idempotency
    if check_exists_in_pinecone(source_filename):
        return 0

    # Add metadata to each chunk
    for doc in docs:
        doc.metadata.update({"source": source_filename, "track": track})

    # Get embeddings and upsert to Pinecone
    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=settings.PINECONE_INDEX_NAME,
    )
    
    logger.info(f"Ingested {len(docs)} chunks from {source_filename}")
    return len(docs)


class PDFIngester:
    """Handles PDF ingestion and chunking for the knowledge base."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load a PDF file and return its pages as documents."""
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Extract metadata from filename
        filename = os.path.basename(pdf_path)
        track_name = extract_track_from_filename(filename)
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata = {
                "source": filename,
                "track": track_name,
                "page": doc.metadata.get("page", 0),
            }
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def ingest_pdf(self, pdf_path: str) -> int:
        """
        Ingest a single PDF file into Pinecone.
        Returns the number of chunks created.
        """
        filename = os.path.basename(pdf_path)
        track_name = extract_track_from_filename(filename)
        
        # Check if already ingested
        if check_exists_in_pinecone(filename):
            logger.info(f"PDF already ingested: {filename} - skipping")
            return 0
        
        # Load and split the PDF
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        
        # Add to Pinecone
        count = ingest_documents(chunks, track_name, filename)
        return count
    
    def ingest_all_pdfs(self) -> dict:
        """
        Ingest all PDFs from the configured directory.
        Returns a summary of the ingestion.
        """
        pdf_dir = Path(settings.PDF_DIR)
        
        if not pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            return {"pdfs_processed": 0, "total_chunks": 0}
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in: {pdf_dir}")
            return {"pdfs_processed": 0, "total_chunks": 0}
        
        total_chunks = 0
        pdfs_processed = 0
        
        for pdf_file in pdf_files:
            try:
                chunks = self.ingest_pdf(str(pdf_file))
                total_chunks += chunks
                pdfs_processed += 1
            except Exception as e:
                logger.error(f"Error ingesting {pdf_file}: {e}")
        
        logger.info(f"Ingestion complete: {pdfs_processed} PDFs, {total_chunks} chunks")
        return {"pdfs_processed": pdfs_processed, "total_chunks": total_chunks}


# Global instance
pdf_ingester = PDFIngester()
