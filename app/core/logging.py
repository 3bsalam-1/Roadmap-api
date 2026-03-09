import logging
import sys
from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("roadmap-api")
    logger.setLevel(log_level)
    
    return logger


logger = setup_logging()
