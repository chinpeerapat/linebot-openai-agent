from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import openai
import os
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()

class VectorStoreCreate(BaseModel):
    name: str

@router.post("/create_store")
async def create_store(store: VectorStoreCreate):
    try:
        vector_store = await client.vector_stores.create(name=store.name)
        return vector_store
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )

@router.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    store_id: str = Form(...),
    purpose: Optional[str] = Form(None)
):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content
        content = await file.read()
        
        # Upload file to vector store
        result = await client.vector_stores.files.create(
            vector_store_id=store_id,
            file=content,
            purpose=purpose or "assistants"
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )

@router.post("/add_file")
async def add_file(store_id: str, file_id: str):
    try:
        if not store_id or not file_id:
            raise HTTPException(
                status_code=400,
                detail="Both store_id and file_id are required"
            )

        result = await client.vector_stores.add_file(
            vector_store_id=store_id,
            file_id=file_id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )

@router.get("/list_files/{store_id}")
async def list_files(store_id: str):
    try:
        files = await client.vector_stores.files.list(vector_store_id=store_id)
        return files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )

@router.get("/retrieve_store/{store_id}")
async def retrieve_store(store_id: str):
    try:
        store = await client.vector_stores.retrieve(vector_store_id=store_id)
        return store
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        ) 