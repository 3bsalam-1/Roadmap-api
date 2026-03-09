from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared rate limiter instance
# Import this in both main.py and route modules to share the same limiter
limiter = Limiter(key_func=get_remote_address)
