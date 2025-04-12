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
    BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
    CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
    CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
    VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # Optional
    MAX_HISTORY_LENGTH = 10
    MAX_FILE_SIZE_MB = 10
    # Performance and feature configurations
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    USE_DIRECT_MEDIA = False  # Changed from True to False
    USE_WEB_SEARCH = True
    USE_FILE_SEARCH = True
    # Conversation state management
    ENABLE_RESPONSE_CHAINING = True  # Use previous_response_id for chaining
    USE_NEW_CONVERSATION_STATE = True  # Use the new ConversationState class
    INTELLIGENT_TRUNCATION = True  # Use intelligent history truncation
    # Default tool configurations
    WEB_SEARCH_CONTEXT_SIZE = os.getenv("WEB_SEARCH_CONTEXT_SIZE", "low")  # Options: low, medium, high
    IMAGE_DETAIL_LEVEL = os.getenv("IMAGE_DETAIL_LEVEL", "high")  # Options: low, high, auto
    # Supported file types based on OpenAI's file search docs
    SUPPORTED_FILE_EXTENSIONS = [
        # Document formats
        ".pdf", ".doc", ".docx", ".pptx", 
        # Programming languages
        ".c", ".cpp", ".cs", ".go", ".java", ".js", ".json", ".php", ".py", ".rb", ".sh", ".ts",
        # Web and styling
        ".html", ".css", 
        # Text formats
        ".txt", ".md", ".tex"
    ]

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
    "OPENAI_API_KEY": Config.API_KEY,
    "CHANNEL_SECRET": Config.CHANNEL_SECRET,
    "CHANNEL_ACCESS_TOKEN": Config.CHANNEL_ACCESS_TOKEN
}

# OPENAI_API_BASE and OPENAI_MODEL have default values, so they're not required

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

# Conversation state management class
class ConversationState:
    def __init__(self):
        self.last_response_id = None
        self.document_references = {}  # Map of doc_id to metadata
        self.last_updated = datetime.now()
        self.first_message = None  # Keep only first system message for context
        
    def update_with_user_message(self, content: str, file_metadata: Optional[dict] = None):
        """Record that a user message was sent, no need to store the actual content"""
        self.last_updated = datetime.now()
        if file_metadata and file_metadata.get("file_id"):
            # Store document reference for future queries
            doc_id = file_metadata.get("file_id")
            self.document_references[doc_id] = {
                "name": file_metadata.get("filename", "Unknown"),
                "timestamp": datetime.now().isoformat(),
                "type": file_metadata.get("type", "unknown")
            }
        
    def update_with_assistant_response(self, response_id: str):
        """Store only the response ID for chaining"""
        self.last_response_id = response_id
        self.last_updated = datetime.now()
        
    def get_document_references(self) -> Dict[str, Dict[str, Any]]:
        """Get all document references associated with this conversation"""
        return self.document_references

# Initialize conversation states
conversation_states: Dict[str, ConversationState] = {}

# Constants and templates
ERROR_MESSAGES = {
    ErrorType.IMAGE_PROCESSING: "❌ รูปภาพไม่ถูกต้อง กรุณาส่งไฟล์ JPEG/PNG\n(Invalid image. Please send JPEG/PNG)",
    ErrorType.PDF_UPLOAD: "❌ ไฟล์เอกสารไม่ถูกต้อง\n(Invalid document file)",
    ErrorType.PDF_UNSUPPORTED: "❌ รูปแบบไฟล์ไม่ได้รับการสนับสนุน กรุณาตรวจสอบรายการไฟล์ที่รองรับ\n(Unsupported file format. Please check supported file types)",
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
    "image": "Briefly describe this image. Respond in user's language.",
    "document": "Summarize this document's key points and content. Respond in user's language."
}

# Helper functions
def update_conversation_history(user_id: str, role: str, content: str, file_content: Optional[dict] = None, response_id: Optional[str] = None):
    """Update the conversation history for a specific user."""
    
    # Use new conversation state management if enabled
    if Config.USE_NEW_CONVERSATION_STATE:
        # Create new state if this is a new user
        if user_id not in conversation_states:
            conversation_states[user_id] = ConversationState()
        
        state = conversation_states[user_id]
        
        # Add message based on role
        if role == "user":
            state.update_with_user_message(content, file_content)
        else:  # assistant
            # If we have a response ID, update that way
            if response_id:
                state.update_with_assistant_response(response_id)
        
        return
        
    # Legacy conversation history management
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    
    message = {"role": role, "content": content}
    
    # Add file metadata without storing the actual file data to save memory
    if file_content:
        # Just store the type of file and any reference ID, not the actual data
        message["file_metadata"] = {
            "type": file_content.get("type", "unknown"),
            "file_id": file_content.get("file_id", None),
            "filename": file_content.get("filename", None)
        }
    
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
    
    try:
        message_content = await line_bot_api.get_message_content(message_id)
        
        # Handle content based on what's returned by the LINE SDK
        if hasattr(message_content, 'content'):
            # message_content.content is a coroutine, not a method
            data = await message_content.content
            total_size = len(data)
            if total_size > size_limit_mb * 1024 * 1024:
                raise ValueError(f"File size exceeds {size_limit_mb}MB limit")
            content.write(data)
        elif hasattr(message_content, 'iter_content'):
            # If iter_content is available
            async for chunk in message_content.iter_content():
                total_size += len(chunk)
                if total_size > size_limit_mb * 1024 * 1024:
                    raise ValueError(f"File size exceeds {size_limit_mb}MB limit")
                content.write(chunk)
        elif hasattr(message_content, '__aiter__'):
            # For older LINE SDK versions with async iterable
            async for chunk in message_content:
                total_size += len(chunk)
                if total_size > size_limit_mb * 1024 * 1024:
                    raise ValueError(f"File size exceeds {size_limit_mb}MB limit")
                content.write(chunk)
        else:
            # Last resort: try to read content as a single blob
            try:
                data = await message_content.read()
                total_size = len(data)
                if total_size > size_limit_mb * 1024 * 1024:
                    raise ValueError(f"File size exceeds {size_limit_mb}MB limit")
                content.write(data)
            except AttributeError:
                # If we can't figure out how to read the content, log detailed info
                print(f"Unknown content type: {type(message_content)}")
                print(f"Available attributes: {dir(message_content)}")
                raise ValueError("Unable to read content from LINE API")
    
        content.seek(0)
        return content, total_size
    except Exception as e:
        print(f"Error processing file content: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

async def process_media_direct(message_id: str, media_type: str) -> Optional[str]:
    """Process media (image or PDF) and return base64 encoded content in the format expected by OpenAI's Responses API."""
    try:
        media_data, size = await process_file_content(message_id)
        print(f"Successfully retrieved {media_type} content, size: {size} bytes")
        
        if media_type == "image":
            # Validate and convert image using PIL
            try:
                image = PIL.Image.open(media_data)
                print(f"Opened image: format={image.format}, mode={image.mode}, size={image.size}")
                
                # Convert to RGB if image is in RGBA mode
                if image.mode in ('RGBA', 'LA'):
                    background = PIL.Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                    print("Converted image with alpha channel to RGB")
                elif image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                    print(f"Converted image from {image.mode} to RGB")
                
                # Resize image if it's too large (OpenAI has limits on image dimensions)
                # High-resolution limit is 768px (short side) x 2000px (long side)
                max_short_side = 768
                max_long_side = 2000
                
                width, height = image.size
                if width > max_long_side or height > max_long_side:
                    # Calculate new dimensions
                    if width >= height:
                        # Width is the long side
                        new_width = min(width, max_long_side)
                        new_height = int(height * (new_width / width))
                        # Ensure short side is not too large
                        if new_height > max_short_side:
                            new_height = max_short_side
                            new_width = int(width * (new_height / height))
                    else:
                        # Height is the long side
                        new_height = min(height, max_long_side)
                        new_width = int(width * (new_height / height))
                        # Ensure short side is not too large
                        if new_width > max_short_side:
                            new_width = max_short_side
                            new_height = int(height * (new_width / width))
                    
                    image = image.resize((new_width, new_height))
                    print(f"Resized image to {new_width}x{new_height} to meet API limits")
                
                # Save the processed image to a BytesIO object as JPEG
                output = BytesIO()
                image.save(output, format='JPEG', quality=95)
                output.seek(0)
                output_size = output.getbuffer().nbytes
                print(f"Converted image to JPEG, new size: {output_size} bytes")
                base64_media = base64.b64encode(output.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_media}"
            except Exception as e:
                print(f"Error processing image with PIL: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        elif media_type == "pdf":
            # Check if the PDF size is within API limits (32MB)
            max_pdf_size_mb = 32  # OpenAI's PDF size limit
            if size > max_pdf_size_mb * 1024 * 1024:
                print(f"PDF size ({size} bytes) exceeds OpenAI's limit of {max_pdf_size_mb}MB")
                return None
                
            base64_media = base64.b64encode(media_data.getvalue()).decode('utf-8')
            print(f"Encoded PDF to base64, size: {len(base64_media)} characters")
            return f"data:application/pdf;base64,{base64_media}"
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
            
    except Exception as e:
        print(f"Error processing media: {str(e)}")
        import traceback
        traceback.print_exc()
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
        
        # Create or get the user's state
        if user_id not in conversation_states:
            conversation_states[user_id] = ConversationState()
        
        state = conversation_states[user_id]
        
        # Prepare API parameters
        api_params = {
            "model": Config.MODEL_NAME,
            "temperature": 1,
            "max_tokens": 2048,
            "store": True  # Always store to enable chaining
        }
        
        # Prepare user message content
        user_message_content = [{"type": "input_text", "text": prompt}]
        
        # Add media content if present
        if media_content:
            if media_content.get("type") == "image" and "base64_data" in media_content:
                user_message_content.append({
                    "type": "input_image",
                    "image_url": media_content["base64_data"],
                    "detail": Config.IMAGE_DETAIL_LEVEL
                })
            elif media_content.get("type") == "pdf" and "base64_data" in media_content:
                user_message_content.append({
                    "type": "input_file",
                    "filename": media_content.get("filename", "document.pdf"),
                    "file_data": media_content["base64_data"]
                })
        
        # Use response chaining when possible
        if state.last_response_id and Config.ENABLE_RESPONSE_CHAINING:
            print(f"Using response chaining for user {user_id}")
            api_params["previous_response_id"] = state.last_response_id
            
            # Simple input when using response chaining
            api_params["input"] = [{"role": "user", "content": user_message_content}]
        else:
            # First message in conversation - add system message
            input_messages = [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a helpful assistant that can communicate in both Thai and English. You always respond concisely in the same language as the user's query. For Thai users, you may provide English translations when appropriate."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_message_content
                }
            ]
            
            # Set full conversation context for first message
            api_params["input"] = input_messages
        
        # Prepare tools
        tools = []
        
        # Add web search tool if enabled
        if Config.USE_WEB_SEARCH:
            tools.append({
                "type": "web_search_preview",
                "search_context_size": Config.WEB_SEARCH_CONTEXT_SIZE
            })

        # Add file search if vector store ID is available and enabled
        if Config.VECTOR_STORE_ID and Config.USE_FILE_SEARCH:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [Config.VECTOR_STORE_ID]
            })
        
        # Add tools only if we have any
        if tools:
            api_params["tools"] = tools
            # Include search results if using file search
            if Config.USE_FILE_SEARCH and Config.VECTOR_STORE_ID:
                api_params["include"] = ["file_search_call.results"]

        # Set timeout to prevent hanging on slow responses
        response = await asyncio.wait_for(
            client.responses.create(**api_params),
            timeout=30.0  # 30 second timeout
        )

        # Extract response text using output_text property
        if hasattr(response, 'output_text'):
            response_text = response.output_text
        else:
            # Handle case where output_text is not available
            for output_item in response.output:
                if output_item.type == "message" and output_item.role == "assistant":
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            response_text = content_item.text
                            break
                    break
            else:
                # If we couldn't find text in the expected structure
                response_text = "Sorry, I couldn't generate a proper response."
        
        # Update state with the new response ID
        if hasattr(response, 'id'):
            # Store only the response ID for future chaining
            state.update_with_assistant_response(response.id)
            print(f"Saved response ID for user {user_id}: {response.id}")
        
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
        import traceback
        traceback.print_exc()
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
    
    # Update user message in conversation state before generating response
    update_conversation_history(user_id, "user", msg)
    
    # Generate response
    response = await generate_text_with_agent(msg, user_id)
    
    # No need to update assistant message history as generate_text_with_agent already stores the response ID
    
    # Send response
    await send_message(user_id, response)

async def handle_image_message(event: MessageEvent):
    """Handle image message events with direct image processing."""
    user_id = event.source.user_id
    
    try:
        print(f"Processing image from user: {user_id}, message ID: {event.message.id}")
        
        # Get base64 encoded image
        base64_image = await process_media_direct(event.message.id, "image")
        
        if not base64_image:
            print(f"Failed to process image for user: {user_id}")
            await send_error(user_id, ErrorType.IMAGE_PROCESSING)
            return
        
        print(f"Successfully processed image for user: {user_id}, base64 length: {len(base64_image)}")
        
        # Update conversation history with a simplified representation
        # We don't store the actual image data in history to save memory
        update_conversation_history(
            user_id, 
            "user", 
            "Uploaded an image [Image analysis requested]",
            {"type": "image"}
        )
            
        # Generate response using the image directly with the Responses API
        response = await generate_text_with_agent(
            PROMPTS["image"],
            user_id,
            {"type": "image", "base64_data": base64_image}
        )
        
        # Send response
        await send_message(user_id, response)
        print(f"Successfully sent response to user: {user_id}")
        
    except Exception as e:
        print(f"Error processing image message: {str(e)}")
        import traceback
        traceback.print_exc()
        await send_error(user_id, ErrorType.IMAGE_PROCESSING)

async def handle_file_message(event: MessageEvent):
    """Handle file message events with vector store for document processing."""
    user_id = event.source.user_id
    file_name = event.message.file_name
    file_size = event.message.file_size
    
    # Check file size first
    if file_size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        await send_error(user_id, ErrorType.FILE_TOO_LARGE)
        return

    # Extract file extension
    file_ext = os.path.splitext(file_name.lower())[1]
    
    # Check if it's a supported file type
    if file_ext in Config.SUPPORTED_FILE_EXTENSIONS:
        try:
            # Ensure we have a vector store ID
            vector_store_id = await get_or_create_vector_store()
            if not vector_store_id:
                await send_error(user_id, ErrorType.VECTOR_STORE)
                return
            
            # Process file through vector store
            file_data, _ = await process_file_content(event.message.id)
            
            # Upload file to OpenAI Files API first
            file_upload = await client.files.create(
                file=file_data,
                purpose="assistants"
            )
            
            # Add file to vector store
            await client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_upload.id
            )
            
            vector_store_file_id = file_upload.id
            
            # Update conversation history
            update_conversation_history(
                user_id,
                "user",
                f"Uploaded document: {file_name} [Document ID: {vector_store_file_id}]",
                {"type": "document", "file_id": vector_store_file_id, "filename": file_name}
            )
            
            # Generate response about the document
            prompt = f"A new document '{file_name}' has been uploaded with ID: {vector_store_file_id}. Please search this document and provide a summary."
            response = await generate_text_with_agent(prompt, user_id)
            
            # Add document reference
            doc_reference = format_document_reference(vector_store_file_id, file_name)
            full_response = f"{response}\n\n{doc_reference}"
            
            # Send response
            await send_message(user_id, full_response)
                
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            await send_error(user_id, ErrorType.PDF_UPLOAD)
    else:
        # Handle unsupported file types
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
    """Health check endpoint with API version and status information."""
    try:
        # Simple check to verify OpenAI API is accessible
        response = await client.models.list()
        api_status = "ok" if response else "error"
    except Exception as e:
        api_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",  # Update this when making significant changes
        "openai_api": {
            "status": api_status,
            "base_url": Config.BASE_URL,
            "model": Config.MODEL_NAME
        },
        "features": {
            "web_search": Config.USE_WEB_SEARCH,
            "file_search": Config.USE_FILE_SEARCH and bool(Config.VECTOR_STORE_ID),
            "image_processing": True,
            "pdf_processing": True,
            "caching": Config.ENABLE_CACHING
        }
    }

# Add cleanup for session on app shutdown
@app.on_event("shutdown")
async def cleanup():
    """Clean up resources on application shutdown."""
    await session.close()

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Validate configuration and setup on application startup."""
    print(f"Starting LINE bot with OpenAI Responses API")
    print(f"Using model: {Config.MODEL_NAME}")
    print(f"Web search enabled: {Config.USE_WEB_SEARCH}")
    print(f"File search enabled: {Config.USE_FILE_SEARCH and bool(Config.VECTOR_STORE_ID)}")
    
    # Check OpenAI API connectivity
    try:
        models = await client.models.list()
        available_models = [model.id for model in models.data]
        print(f"Available OpenAI models: {', '.join(available_models[:5])}...")
        
        if Config.MODEL_NAME not in available_models:
            print(f"WARNING: Configured model {Config.MODEL_NAME} not found in available models")
            print(f"Available models that support responses API may include: gpt-4o, gpt-4o-mini")
    except Exception as e:
        print(f"ERROR: Failed to connect to OpenAI API: {e}")
        # We don't want to fail startup completely, so just log the error
    
    # Check vector store if enabled
    if Config.USE_FILE_SEARCH and Config.VECTOR_STORE_ID:
        try:
            vector_store = await client.vector_stores.retrieve(Config.VECTOR_STORE_ID)
            print(f"Using vector store: {vector_store.id} ({vector_store.name})")
        except Exception as e:
            print(f"ERROR: Failed to connect to vector store: {e}")
            Config.USE_FILE_SEARCH = False
            print("Disabling file search due to vector store error")
    
    print("Startup complete.")