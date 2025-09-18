import os
import uuid
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "target"), exist_ok=True)

def save_upload_file(upload_file: UploadFile, is_target=False) -> tuple:
    file_id = str(uuid.uuid4())
    extension = upload_file.filename.split(".")[-1]
    if not is_target:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")
    else:
        file_path = os.path.join(UPLOAD_DIR, "target", f"{file_id}.{extension}")

    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_id, file_path
