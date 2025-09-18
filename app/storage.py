import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
S3_BUCKET = os.getenv("S3_BUCKET", "images")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_KEY")

def save_image(image_id: str, filename: str, content: bytes) -> str:
    if STORAGE_BACKEND == "s3":
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_KEY,
            aws_secret_access_key=S3_SECRET,
        )
        key = f"{image_id}_{filename}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content)
        return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
    else:
        os.makedirs("data/images", exist_ok=True)
        path = f"data/images/{image_id}_{filename}"
        with open(path, "wb") as f:
            f.write(content)
        return f"http://127.0.0.1:8000/data/images/{image_id}_{filename}"
