"""
API Key Authentication & Security Middleware/Dependencies for SatQuery AI Agent Registry.
Enforces admin-level privileges for registration, updates, and deregistration.
Allows router-level privileges for agent discovery and health check reporting.
"""

import os
from fastapi import Header, HTTPException, status, Security
from fastapi.security import APIKeyHeader

ADMIN_API_KEY = os.getenv("SATQUERY_ADMIN_KEY", "satquery-admin-key-2026")
ROUTER_API_KEY = os.getenv("SATQUERY_ROUTER_KEY", "satquery-router-key-2026")

api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_admin_key(x_api_key: str = Security(api_key_header_scheme)):
    """
    Requires Admin API Key for mutating endpoints (POST /agents/register, PUT, DELETE).
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required security header 'X-API-Key'."
        )
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Admin API Key. Administrative privileges required for this operation."
        )
    return x_api_key


def verify_read_key(x_api_key: str = Security(api_key_header_scheme)):
    """
    Allows Admin or Router API Key for discovery and health endpoints.
    If no key is provided, permits read access for public discovery, but if key is provided, validates it.
    """
    if x_api_key is not None:
        if x_api_key not in [ADMIN_API_KEY, ROUTER_API_KEY]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key provided."
            )
    return x_api_key
