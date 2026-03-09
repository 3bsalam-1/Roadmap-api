import cloudinary
import cloudinary.uploader
import cloudinary.api
import httpx
from app.core.config import settings
from app.core.logging import logger

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
)


def upload_pdf_to_cloudinary(file_bytes: bytes, filename: str) -> str:
    """
    Upload PDF bytes to Cloudinary. Returns the secure URL.
    
    Args:
        file_bytes: PDF file content as bytes
        filename: Name of the file (e.g., "machine_learning.pdf")
        
    Returns:
        The secure URL of the uploaded PDF
    """
    logger.info(f"Uploading {filename} to Cloudinary")
    
    result = cloudinary.uploader.upload(
        file_bytes,
        resource_type="raw",  # PDFs must use resource_type="raw"
        public_id=f"roadmaps/{filename.replace('.pdf', '')}",
        overwrite=False,  # Don't overwrite existing PDFs
        format="pdf",
    )
    
    logger.info(f"Uploaded to Cloudinary: {result['secure_url']}")
    return result["secure_url"]


async def download_pdf_from_url(url: str) -> bytes:
    """
    Download a PDF from any URL (including Cloudinary) and return bytes.
    
    Args:
        url: URL to download the PDF from
        
    Returns:
        PDF content as bytes
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content


def list_cloudinary_pdfs() -> list[dict]:
    """
    List all PDFs stored in the roadmaps/ Cloudinary folder.
    
    Returns:
        List of dicts with 'name' and 'url' keys
    """
    try:
        result = cloudinary.api.resources(
            type="upload",
            resource_type="raw",
            prefix="roadmaps/",
            max_results=100,
        )
        return [{"name": r["public_id"], "url": r["secure_url"]} for r in result["resources"]]
    except Exception as e:
        logger.error(f"Error listing Cloudinary PDFs: {e}")
        return []


def delete_cloudinary_pdf(public_id: str) -> bool:
    """
    Delete a PDF from Cloudinary.
    
    Args:
        public_id: The public ID of the file (e.g., "roadmaps/machine_learning")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cloudinary.uploader.destroy(public_id, resource_type="raw")
        logger.info(f"Deleted {public_id} from Cloudinary")
        return True
    except Exception as e:
        logger.error(f"Error deleting {public_id} from Cloudinary: {e}")
        return False
