#!/usr/bin/env python3
"""
One-time script to ingest all PDFs from the data/pdfs directory into ChromaDB.
Can be re-run safely - it skips PDFs that are already ingested.
"""

import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.ingestor import pdf_ingester
from app.core.logging import logger


def main():
    """Ingest all PDFs and print summary."""
    logger.info("Starting PDF ingestion...")
    
    try:
        result = pdf_ingester.ingest_all_pdfs()
        
        print("\n" + "=" * 50)
        print("PDF INGESTION SUMMARY")
        print("=" * 50)
        print(f"PDFs processed: {result['pdfs_processed']}")
        print(f"Total chunks stored: {result['total_chunks']}")
        print("=" * 50)
        
        if result['pdfs_processed'] == 0:
            print("\nNo PDFs were ingested. Please add PDF files to:")
            print("  ./data/pdfs/")
            print("\nPDF naming convention: snake_case")
            print("  e.g., machine_learning.pdf, web_development.pdf")
        
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
