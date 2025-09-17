# app/utils/storage.py
import boto3
import os
import shutil
from urllib.parse import urljoin

S3_BUCKET = os.getenv("S3_BUCKET", "image-bucket")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", None)  # for MinIO
UPLOAD_MODE = os.getenv("UPLOAD_MODE", "s3")  # "s3" or "local"
LOCAL_UPLOAD_DIR = os.getenv("LOCAL_UPLOAD_DIR", "uploads")

s3 = boto3.client('s3',
                  endpoint_url=S3_ENDPOINT,
                  aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                  aws_secret_access_key=os.getenv("S3_SECRET_KEY"))

def upload_file(filepath, key):
    """
    Uploads a file to S3 or local storage depending on UPLOAD_MODE.
    Returns the URL or path to the uploaded file.
    """
    if UPLOAD_MODE == "local":
        # Ensure local upload directory exists
        dest_path = os.path.join(LOCAL_UPLOAD_DIR, key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(filepath, dest_path)
        # Return local file path or URL
        # If serving via static files, you may want to return a URL instead
        return dest_path
    else:
        s3.upload_file(filepath, S3_BUCKET, key)
        # construct URL - in production use signed URLs or CDN
        if S3_ENDPOINT:
            return urljoin(S3_ENDPOINT, f"{S3_BUCKET}/{key}")
        return f"s3://{S3_BUCKET}/{key}"
