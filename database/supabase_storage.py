"""
Supabase Storage Client for SatQuery AI.
Handles direct binary file uploads and public URL resolution against Supabase Storage buckets.
"""

import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Configurable environment variables with standard fallbacks
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://gqqznwdujwpuvmllmwgv.supabase.co").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "satellite-images")


class SupabaseStorageClient:
    """
    Direct REST API client for Supabase Storage.
    Uploads binary imagery (GeoTIFF, PNG, JPEG) and generates public storage URLs.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        raw_url = supabase_url or SUPABASE_URL
        if "/rest/v1" in raw_url:
            raw_url = raw_url.split("/rest/v1")[0]
        self.supabase_url = raw_url.rstrip("/")
        self.supabase_key = supabase_key or SUPABASE_KEY
        self.bucket_name = bucket_name or SUPABASE_BUCKET

    def upload_image(
        self,
        file_bytes: bytes,
        destination_path: str,
        content_type: str = "image/tiff"
    ) -> Dict[str, Any]:
        """
        Uploads raw binary bytes to the Supabase Storage bucket.
        Path format: satellite-images/{user_id}/{image_id}_{filename}
        Returns storage path and public access URL.
        """
        public_url = f"{self.supabase_url}/storage/v1/object/public/{self.bucket_name}/{destination_path}"
        storage_path = f"{self.bucket_name}/{destination_path}"

        # If API key is available, perform real HTTP POST upload to Supabase Storage
        if self.supabase_key:
            upload_url = f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{destination_path}"
            headers = {
                "Authorization": f"Bearer {self.supabase_key}",
                "apiKey": self.supabase_key,
                "Content-Type": content_type,
                "x-upsert": "true"
            }

            response = requests.post(upload_url, data=file_bytes, headers=headers, timeout=30)
            
            # If bucket is not found (404), attempt automatic bucket creation
            if response.status_code == 404 or "not found" in response.text.lower():
                self._create_bucket_if_missing()
                response = requests.post(upload_url, data=file_bytes, headers=headers, timeout=30)

            if response.status_code not in (200, 201):
                logger.error(f"Supabase Storage Upload Error ({response.status_code}): {response.text}")
                raise RuntimeError(
                    f"Failed to upload object to Supabase Storage ({response.status_code}): {response.text}"
                )

        return {
            "bucket": self.bucket_name,
            "storage_path": storage_path,
            "public_url": public_url
        }

    def _create_bucket_if_missing(self) -> bool:
        """Creates the bucket if it doesn't already exist on Supabase."""
        if not self.supabase_key:
            return False
        create_url = f"{self.supabase_url}/storage/v1/bucket"
        headers = {
            "Authorization": f"Bearer {self.supabase_key}",
            "apiKey": self.supabase_key,
            "Content-Type": "application/json"
        }
        payload = {
            "id": self.bucket_name,
            "name": self.bucket_name,
            "public": True
        }
        res = requests.post(create_url, json=payload, headers=headers, timeout=15)
        return res.status_code in (200, 201)
