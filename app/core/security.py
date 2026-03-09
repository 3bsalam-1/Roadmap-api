from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_admin(api_key: str = Security(api_key_header)):
    """
    FastAPI dependency — raises 403 if X-API-Key header is missing or wrong.
    
    Usage:
        @router.post("/endpoint", dependencies=[Depends(require_admin)])
        async def my_endpoint():
            ...
    """
    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
