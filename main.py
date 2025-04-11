import os
import sys
import asyncio
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import base64
from datetime import datetime
from enum import Enum

import aiohttp
import PIL.Image
from fastapi import Request, FastAPI, HTTPException, BackgroundTasks
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, trace, WebSearchTool, FileSearchTool
from openai import OpenAI
    
from linebot.models import (
    MessageEvent, TextSendMessage, ImageMessage, FileMessage
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
class Config:
    BASE_URL = os.getenv("EXAMPLE_BASE_URL")
    API_KEY = os.getenv("EXAMPLE_API_KEY")
    MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME")
    CHANNEL_SECRET = os.getenv("ChannelSecret")
    CHANNEL_ACCESS_TOKEN = os.getenv("ChannelAccessToken")
    VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # Optional
    MAX_HISTORY_LENGTH = 10
    MAX_FILE_SIZE_MB = 10
    # New configurations for perf optimizations
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    USE_DIRECT_MEDIA = True

# Define error types as enum for better type safety
class ErrorType(str, Enum):
    IMAGE_PROCESSING = "image_processing"
    PDF_UPLOAD = "pdf_upload"
    PDF_UNSUPPORTED = "pdf_unsupported"
    VECTOR_STORE = "vector_store"
    GENERAL = "general"
    FILE_TOO_LARGE = "file_too_large"
    INVALID_FORMAT = "invalid_format"
    UNSUPPORTED_MESSAGE_TYPE = "unsupported_message_type"

# Validate required environment variables
required_vars = {
    "EXAMPLE_BASE_URL": Config.BASE_URL,
    "EXAMPLE_API_KEY": Config.API_KEY,
    "EXAMPLE_MODEL_NAME": Config.MODEL_NAME,
    "ChannelSecret": Config.CHANNEL_SECRET,
    "ChannelAccessToken": Config.CHANNEL_ACCESS_TOKEN
}

missing_vars = [var_name for var_name, var_value in required_vars.items() if not var_value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI()

# Initialize aiohttp session
session = aiohttp.ClientSession()

# Initialize LINE bot API client
http_client = AiohttpAsyncHttpClient(session)
parser = WebhookParser(Config.CHANNEL_SECRET)
line_bot_api = AsyncLineBotApi(Config.CHANNEL_ACCESS_TOKEN, http_client)

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url=Config.BASE_URL,
    api_key=Config.API_KEY,
)

# Initialize conversation histories
conversation_histories: Dict[str, List[Dict[str, Any]]] = {}

# Initialize response cache
response_cache: Dict[str, Tuple[str, float]] = {}  # (response, timestamp)

# Constants and templates
ERROR_MESSAGES = {
    ErrorType.IMAGE_PROCESSING: "❌ รูปภาพไม่ถูกต้อง กรุณาส่งไฟล์ JPEG/PNG\n(Invalid image. Please send JPEG/PNG)",
    ErrorType.PDF_UPLOAD: "❌ ไฟล์ PDF ไม่ถูกต้อง\n(Invalid PDF file)",
    ErrorType.PDF_UNSUPPORTED: "❌ รองรับเฉพาะไฟล์ PDF\n(PDF files only)",
    ErrorType.VECTOR_STORE: "❌ ไม่สามารถจัดเก็บเอกสารได้\n(Unable to store document)",
    ErrorType.GENERAL: "❌ เกิดข้อผิดพลาด กรุณาลองใหม่\n(Error occurred. Please try again)",
    ErrorType.FILE_TOO_LARGE: "❌ ไฟล์ใหญ่เกิน 10MB\n(File exceeds 10MB)",
    ErrorType.INVALID_FORMAT: "❌ รูปแบบไฟล์ไม่ถูกต้อง\n(Invalid file format)",
    ErrorType.UNSUPPORTED_MESSAGE_TYPE: "❌ ไม่รองรับประเภทข้อความนี้\n(Unsupported message type)"
}

DOCUMENT_REFERENCE = """
📄 ข้อมูลอ้างอิงเอกสาร | Document Reference
ID: {doc_id}
ชื่อ | Name: {doc_name}
อัปโหลดเมื่อ | Uploaded: {timestamp}

คุณสามารถอ้างถึงเอกสารนี้ในคำถามต่อไปได้โดย:
You can refer to this document in future questions using:

1️⃣ "แสดงข้อมูลเกี่ยวกับเอกสาร {doc_id}"
   "Show me information about document {doc_id}"

2️⃣ "เอกสาร {doc_id} พูดถึงอะไรเกี่ยวกับ[หัวข้อ]?"
   "What does document {doc_id} say about [your topic]?"
"""

PROMPTS = {
    "image": "Describe this image scientifically. Respond in user's language.",
    "pdf": "Summarize this PDF document's key points and content. Respond in user's language."
}

# Helper functions
def update_conversation_history(user_id: str, role: str, content: str, file_content: Optional[dict] = None):
    """Update the conversation history for a specific user."""
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    
    message = {"role": role, "content": content}
    if file_content:
        message["file"] = file_content
    
    conversation_histories[user_id].append(message)
    
    # Keep only the last MAX_HISTORY_LENGTH messages
    if len(conversation_histories[user_id]) > Config.MAX_HISTORY_LENGTH:
        conversation_histories[user_id] = conversation_histories[user_id][-Config.MAX_HISTORY_LENGTH:]

def format_document_reference(doc_id: str, doc_name: str) -> str:
    """Format document reference information in a consistent way with both Thai and English."""
    # Clean and truncate document name if too long
    safe_doc_name = doc_name[:50] + "..." if len(doc_name) > 50 else doc_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return DOCUMENT_REFERENCE.format(
        doc_id=doc_id,
        doc_name=safe_doc_name,
        timestamp=timestamp
    )

async def send_message(user_id: str, message: str) -> bool:
    """Send a text message to a user."""
    try:
        await line_bot_api.push_message(user_id, TextSendMessage(text=message))
        return True
    except Exception as e:
        print(f"Failed to send message to user {user_id}: {e}")
        return False

async def send_error(user_id: str, error_type: ErrorType) -> bool:
    """Send an error message to a user."""
    error_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES[ErrorType.GENERAL])
    return await send_message(user_id, error_message)

# Caching mechanism
def get_cache_entry(cache_key: str) -> Optional[str]:
    """Get a response from cache if it exists and is not expired."""
    if not Config.ENABLE_CACHING:
        return None
        
    if cache_key in response_cache:
        response, timestamp = response_cache[cache_key]
        # Check if cache entry is still valid
        if time.time() - timestamp < Config.CACHE_TTL_SECONDS:
            return response
    return None

def set_cache_entry(cache_key: str, response: str):
    """Set a response in the cache with current timestamp."""
    if not Config.ENABLE_CACHING:
        return
        
    response_cache[cache_key] = (response, time.time())

# Direct media processing
async def process_file_content(message_id: str, size_limit_mb: int = Config.MAX_FILE_SIZE_MB) -> Tuple[BytesIO, int]:
    """Process file content from LINE API and return BytesIO object and size."""
    content = BytesIO()
    total_size = 0
    
    message_content = await line_bot_api.get_message_content(message_id)
    async for chunk in message_content:
        total_size += len(chunk)
        if total_size > size_limit_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds {size_limit_mb}MB limit")
        content.write(chunk)
    
    content.seek(0)
    return content, total_size

async def process_media_direct(message_id: str, media_type: str) -> Optional[str]:
    """Process media (image or PDF) and return base64 encoded content."""
    try:
        media_data, _ = await process_file_content(message_id)
        
        if media_type == "image":
            # Validate and convert image using PIL
            try:
                image = PIL.Image.open(media_data)
                # Convert to RGB if image is in RGBA mode
                if image.mode in ('RGBA', 'LA'):
                    background = PIL.Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                # Save the processed image to a BytesIO object as JPEG
                output = BytesIO()
                image.save(output, format='JPEG', quality=95)
                output.seek(0)
                base64_media = base64.b64encode(output.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_media}"
            except Exception as e:
                print(f"Error processing image with PIL: {e}")
                return None
        elif media_type == "pdf":
            base64_media = base64.b64encode(media_data.getvalue()).decode('utf-8')
            return f"data:application/pdf;base64,{base64_media}"
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
            
    except Exception as e:
        print(f"Error processing media: {e}")
        return None

# Legacy vector store methods (kept for fallback)
async def get_or_create_vector_store() -> Optional[str]:
    """Get existing vector store ID or create a new one."""
    try:
        if Config.VECTOR_STORE_ID:
            return Config.VECTOR_STORE_ID
        
        # Create a new vector store
        store = await client.vector_stores.create(
            name=f"line_bot_store_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Update config
        Config.VECTOR_STORE_ID = store.id
        os.environ["VECTOR_STORE_ID"] = store.id
        print(f"Created new vector store with ID: {store.id}")
        
        return store.id
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

async def list_stored_documents(user_id: str) -> str:
    """List all documents stored in the vector store."""
    try:
        vector_store_id = Config.VECTOR_STORE_ID
        if not vector_store_id:
            return "No documents stored yet."
            
        files = await client.vector_stores.files.list(vector_store_id=vector_store_id)
        
        if not files.data:
            return "No documents found in storage."
            
        doc_list = ["📚 Stored Documents:"]
        for file in files.data:
            doc_list.append(f"\n📄 ID: {file.id}")
            doc_list.append(f"📝 Name: {file.filename}")
            doc_list.append(f"📅 Created: {file.created_at}")
            doc_list.append("---")
            
        return "\n".join(doc_list)
        
    except Exception as e:
        print(f"Error listing documents: {e}")
        return "Unable to list documents at this time."

# Optimized text generation
async def generate_text_with_agent(prompt: str, user_id: str, media_content: Optional[dict] = None) -> str:
    """Generate text response using OpenAI with conversation history and context."""
    try:
        # Check cache for simple text queries
        if not media_content and Config.ENABLE_CACHING:
            cache_key = f"{hash(prompt)}_{user_id[:8]}"
            cached_response = get_cache_entry(cache_key)
            if cached_response:
                print(f"Cache hit for user {user_id}")
                return cached_response
        
        # Prepare system message
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are a helpful assistant that can communicate in both Thai and English. You always respond concisely in the same language as the user's query. For Thai users, you may provide English translations when appropriate."
                }
            ]
        }
        
        # Prepare user message with content
        user_message_content = [{"type": "input_text", "text": prompt}]
        
        # Add media content if present
        if media_content:
            if media_content.get("type") == "image" and "base64_data" in media_content:
                user_message_content.append({
                    "type": "input_image",
                    "image_url": media_content["base64_data"]
                })
            elif media_content.get("type") == "pdf" and "base64_data" in media_content:
                user_message_content.append({
                    "type": "input_file",
                    "filename": media_content.get("filename", "document.pdf"),
                    "file_data": media_content["base64_data"]
                })
        
        user_message = {
            "role": "user",
            "content": user_message_content
        }
        
        # Get conversation history
        history = conversation_histories.get(user_id, [])
        
        # Prepare full message list
        messages = [system_message]
        
        # Add simplified conversation history (limited to reduce tokens)
        for msg in history[-5:]:  # Only add last 5 messages to reduce context size
            if msg["role"] == "user":
                messages.append({"role": "user", "content": [{"type": "input_text", "text": msg["content"]}]})
            else:
                messages.append({"role": "assistant", "content": [{"type": "input_text", "text": msg["content"]}]})
        
        # Add current user message
        messages.append(user_message)

        # Prepare tools
        tools = [
            {
                "type": "web_search_preview",
                "search_context_size": "low"
            }
        ]

        # Add file search if vector store ID is available and using vector store approach
        if Config.VECTOR_STORE_ID and not Config.USE_DIRECT_MEDIA:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [Config.VECTOR_STORE_ID]
            })

        # Set timeout to prevent hanging on slow responses
        response = await asyncio.wait_for(
            client.responses.create(
                model=Config.MODEL_NAME,
                input=messages,
                text={
                    "format": {
                        "type": "text"
                    }
                },
                reasoning={},
                tools=tools,
                temperature=1,
                max_output_tokens=2048,
                top_p=1,
                store=True
            ),
            timeout=30.0  # 30 second timeout
        )

        # Extract response text based on response type
        if hasattr(response, 'output_text'):
            response_text = response.output_text  # For text & websearch responses
        else:
            response_text = response.text.value  # For file search responses
        
        # Cache response for text-only queries
        if not media_content and Config.ENABLE_CACHING:
            cache_key = f"{hash(prompt)}_{user_id[:8]}"
            set_cache_entry(cache_key, response_text)

        return response_text

    except asyncio.TimeoutError:
        print(f"Response generation timed out for user {user_id}")
        return "I'm sorry, but it's taking longer than expected to process your request. Please try again with a simpler query."
    except Exception as e:
        print(f"Error generating text: {e}")
        return ERROR_MESSAGES[ErrorType.GENERAL]

# Message handlers with timing metrics
async def handle_message_with_timing(handler_func, event):
    """Measure and log handler execution time."""
    start_time = time.time()
    result = await handler_func(event)
    elapsed_time = time.time() - start_time
    print(f"{handler_func.__name__} took {elapsed_time:.2f} seconds")
    return result

async def handle_text_message(event: MessageEvent):
    """Handle text message events."""
    user_id = event.source.user_id
    msg = event.message.text
    
    # Generate response
    response = await generate_text_with_agent(msg, user_id)
    
    # Update conversation history
    update_conversation_history(user_id, "user", msg)
    update_conversation_history(user_id, "assistant", response)
    
    # Send response
    await send_message(user_id, response)

async def handle_image_message(event: MessageEvent):
    """Handle image message events with direct image processing."""
    user_id = event.source.user_id
    
    try:
        # Get base64 encoded image
        base64_image = await process_media_direct(event.message.id, "image")
        
        if not base64_image:
            await send_error(user_id, ErrorType.IMAGE_PROCESSING)
            return
            
        # Generate response using the image directly
        response = await generate_text_with_agent(
            PROMPTS["image"],
            user_id,
            {"type": "image", "base64_data": base64_image}
        )
        
        # Update conversation history and send response
        update_conversation_history(user_id, "user", "Uploaded an image", {"type": "image"})
        update_conversation_history(user_id, "assistant", response)
        
        # Send response
        await send_message(user_id, response)
        
    except Exception as e:
        print(f"Error processing image message: {e}")
        await send_error(user_id, ErrorType.IMAGE_PROCESSING)

async def handle_file_message(event: MessageEvent):
    """Handle file message events with direct PDF processing."""
    user_id = event.source.user_id
    file_name = event.message.file_name
    file_size = event.message.file_size
    
    # Check file size first
    if file_size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        await send_error(user_id, ErrorType.FILE_TOO_LARGE)
        return

    # Check if it's a PDF file
    if file_name.lower().endswith('.pdf'):
        try:
            if Config.USE_DIRECT_MEDIA:
                # Process PDF directly with base64
                base64_pdf = await process_media_direct(event.message.id, "pdf")
                
                if not base64_pdf:
                    await send_error(user_id, ErrorType.PDF_UPLOAD)
                    return
                
                # Generate response using the PDF directly
                response = await generate_text_with_agent(
                    PROMPTS["pdf"],
                    user_id,
                    {"type": "pdf", "base64_data": base64_pdf, "filename": file_name}
                )
                
                # Update conversation history
                update_conversation_history(
                    user_id,
                    "user",
                    f"Uploaded PDF: {file_name}",
                    {"type": "pdf"}
                )
                
                # Add document reference without vector store ID
                doc_reference = f"📄 Document: {file_name}\n📅 Uploaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                full_response = f"{response}\n\n{doc_reference}"
                
                update_conversation_history(user_id, "assistant", full_response)
                await send_message(user_id, full_response)
            else:
                # Use legacy vector store approach as fallback
                vector_store_id = await get_or_create_vector_store()
                if not vector_store_id:
                    await send_error(user_id, ErrorType.VECTOR_STORE)
                    return
                
                # Process PDF through vector store
                pdf_data, _ = await process_file_content(event.message.id)
                
                # Upload file to vector store
                file_upload = await client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file=pdf_data,
                    purpose="assistants"
                )
                
                # Add file to vector store
                await client.vector_stores.add_file(
                    vector_store_id=vector_store_id,
                    file_id=file_upload.id
                )
                
                vector_store_file_id = file_upload.id
                
                # Update conversation history
                update_conversation_history(
                    user_id,
                    "user",
                    f"Uploaded PDF: {file_name}",
                    {"type": "pdf", "file_id": vector_store_file_id}
                )
                
                # Generate response about the PDF
                prompt = f"A new PDF document has been uploaded with ID: {vector_store_file_id}. Please search this document and provide a summary."
                response = await generate_text_with_agent(prompt, user_id)
                
                # Add document reference
                doc_reference = format_document_reference(vector_store_file_id, file_name)
                full_response = f"{response}\n\n{doc_reference}"
                
                update_conversation_history(user_id, "assistant", full_response)
                await send_message(user_id, full_response)
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            await send_error(user_id, ErrorType.PDF_UPLOAD)
    else:
        # Handle non-PDF files
        await send_error(user_id, ErrorType.PDF_UNSUPPORTED)

async def process_line_event(event: MessageEvent):
    """Process LINE event using a dispatcher pattern with timing."""
    user_id = event.source.user_id
    print(f"Processing {event.message.type} message from user: {user_id}")
    
    # Message type handlers
    handlers = {
        "text": handle_text_message,
        "image": handle_image_message,
        "file": handle_file_message
    }
    
    # Get the appropriate handler
    handler = handlers.get(event.message.type)
    if not handler:
        await send_error(user_id, ErrorType.UNSUPPORTED_MESSAGE_TYPE)
        return
        
    # Execute the handler with timing
    try:
        await handle_message_with_timing(handler, event)
    except Exception as e:
        print(f"Error in {event.message.type} handler: {e}")
        await send_error(user_id, ErrorType.GENERAL)

# Process events in parallel
async def process_events_parallel(events):
    """Process multiple events in parallel."""
    tasks = []
    for event in events:
        if isinstance(event, MessageEvent):
            tasks.append(process_line_event(event))
    
    if tasks:
        await asyncio.gather(*tasks)

# FastAPI endpoints
@app.post("/")
async def handle_callback(request: Request, background_tasks: BackgroundTasks):
    """Handle LINE webhook callbacks."""
    signature = request.headers.get('X-Line-Signature', '')
    
    # Get request body as text
    body = await request.body()
    body_text = body.decode()

    try:
        events = parser.parse(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Process events in parallel for better throughput
    background_tasks.add_task(process_events_parallel, events)
    
    # Return 200 OK immediately
    return {"message": "OK"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Add cleanup for session on app shutdown
@app.on_event("shutdown")
async def cleanup():
    """Clean up resources on application shutdown."""
    await session.close()