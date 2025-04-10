import os
import sys
import asyncio
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import base64
from datetime import datetime

import aiohttp
import PIL.Image
from fastapi import Request, FastAPI, HTTPException, BackgroundTasks
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, trace, WebSearchTool, FileSearchTool

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
BASE_URL = os.getenv("EXAMPLE_BASE_URL")
API_KEY = os.getenv("EXAMPLE_API_KEY")
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME")
CHANNEL_SECRET = os.getenv("ChannelSecret")
CHANNEL_ACCESS_TOKEN = os.getenv("ChannelAccessToken")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # Optional

# Validate required environment variables
required_vars = {
    "EXAMPLE_BASE_URL": BASE_URL,
    "EXAMPLE_API_KEY": API_KEY,
    "EXAMPLE_MODEL_NAME": MODEL_NAME,
    "ChannelSecret": CHANNEL_SECRET,
    "ChannelAccessToken": CHANNEL_ACCESS_TOKEN
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
parser = WebhookParser(CHANNEL_SECRET)
line_bot_api = AsyncLineBotApi(CHANNEL_ACCESS_TOKEN, http_client)

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Initialize conversation histories
conversation_histories: Dict[str, List[Dict[str, Any]]] = {}

# Maximum conversation history length
MAX_HISTORY_LENGTH = 10

# Constants
MAX_FILE_SIZE_MB = 10

# Error messages
ERROR_MESSAGES = {
    "image_processing": "❌ รูปภาพไม่ถูกต้อง กรุณาส่งไฟล์ JPEG/PNG\n(Invalid image. Please send JPEG/PNG)",
    "pdf_upload": "❌ ไฟล์ PDF ไม่ถูกต้อง\n(Invalid PDF file)",
    "pdf_unsupported": "❌ รองรับเฉพาะไฟล์ PDF\n(PDF files only)",
    "vector_store": "❌ ไม่สามารถจัดเก็บเอกสารได้\n(Unable to store document)",
    "general": "❌ เกิดข้อผิดพลาด กรุณาลองใหม่\n(Error occurred. Please try again)",
    "file_too_large": "❌ ไฟล์ใหญ่เกิน 10MB\n(File exceeds 10MB)",
    "invalid_format": "❌ รูปแบบไฟล์ไม่ถูกต้อง\n(Invalid file format)",
    "unsupported_message_type": "❌ ไม่รองรับประเภทข้อความนี้\n(Unsupported message type)"
}

# Document reference format
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

# Prompts
PROMPTS = {
    "image": "Describe this image scientifically. Respond in user's language.",
    "pdf": "Summarize this PDF document's key points and content. Respond in user's language."
}

# Initialize the agent with proper tools
agent = Agent(
    name="LINE Bot Assistant",
    instructions="""Bilingual assistant (Thai/English). Match user's language. Use web search for current info and file_search for Document IDs. Be concise yet informative.""",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=[VECTOR_STORE_ID] if VECTOR_STORE_ID else [],
            include_search_results=True,
        )
    ],
)

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
    if len(conversation_histories[user_id]) > MAX_HISTORY_LENGTH:
        conversation_histories[user_id] = conversation_histories[user_id][-MAX_HISTORY_LENGTH:]

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

async def send_message(user_id: str, message: str):
    """Send a text message to a user."""
    try:
        await line_bot_api.push_message(user_id, TextSendMessage(text=message))
        return True
    except Exception as e:
        print(f"Failed to send message to user {user_id}: {e}")
        return False

async def send_error(user_id: str, error_type: str):
    """Send an error message to a user."""
    error_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general"])
    return await send_message(user_id, error_message)

async def process_file_content(message_id: str, size_limit_mb: int = MAX_FILE_SIZE_MB) -> Tuple[BytesIO, int]:
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

async def process_image(message_id: str) -> dict:
    """Process an image message and return content for vision model."""
    try:
        image_data, _ = await process_file_content(message_id)
        
        # Convert to PIL Image for potential preprocessing
        image = PIL.Image.open(image_data)
        
        # Convert image to base64 for OpenAI vision model
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            }
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

async def process_pdf(message_id: str, file_name: str) -> Tuple[str, Optional[str]]:
    """Process a PDF file and upload to vector store."""
    try:
        pdf_data, file_size = await process_file_content(message_id)
        
        # Upload to OpenAI's vector store
        file_upload = await client.files.create(
            file=pdf_data,
            purpose="assistants",
            file_name=file_name
        )
        
        vector_store_file_id = file_upload.id
        print(f"Successfully uploaded PDF to vector store with ID: {vector_store_file_id}")
        return "Success", vector_store_file_id
            
    except ValueError as ve:
        # File size exception
        return await send_error("file_too_large"), None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return await send_error("pdf_upload"), None

async def generate_text_with_agent(prompt: str, user_id: str, content: Optional[dict] = None):
    """Generate text response using the agent with conversation history and context."""
    try:
        # Get conversation history
        history = conversation_histories.get(user_id, [])
        
        # Build conversation context
        if history:
            conversation_context = "Previous conversation:\n" + "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in history
            )
        else:
            conversation_context = ""

        # Add file context if available
        if content and content.get("type") == "pdf":
            file_id = content.get("file_id")
            if file_id:
                full_prompt = f"{conversation_context}\nCurrent request: {prompt}\nDocument ID to search: {file_id}"
            else:
                full_prompt = f"{conversation_context}\nCurrent request: {prompt}"
        else:
            full_prompt = f"{conversation_context}\nCurrent request: {prompt}"

        # Run the agent with tracing
        with trace(f"Generate response for user {user_id}"):
            result = await Runner.run(agent, full_prompt)
            response = result.final_output

            # Log tool usage if any
            if result.new_items:
                print("\n".join([str(out) for out in result.new_items]))

        return response

    except Exception as e:
        print(f"Error generating text: {e}")
        return ERROR_MESSAGES["general"]

# Message handlers
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
    """Handle image message events."""
    user_id = event.source.user_id
    
    try:
        # Process the image
        image_content = await process_image(event.message.id)
        
        # Generate response using the image
        response = await generate_text_with_agent(PROMPTS["image"], user_id, image_content)
        
        # Update conversation history
        update_conversation_history(user_id, "user", "Uploaded an image", {"type": "image"})
        update_conversation_history(user_id, "assistant", response)
        
        # Send response
        await send_message(user_id, response)
        
    except Exception as e:
        print(f"Error processing image message: {e}")
        await send_error(user_id, "image_processing")

async def handle_file_message(event: MessageEvent):
    """Handle file message events."""
    user_id = event.source.user_id
    file_name = event.message.file_name
    file_size = event.message.file_size
    
    # Check file size first
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await send_error(user_id, "file_too_large")
        return

    # Check if it's a PDF file
    if file_name.lower().endswith('.pdf'):
        try:
            # Process the PDF
            status_message, vector_store_file_id = await process_pdf(event.message.id, file_name)
            
            if vector_store_file_id:
                # Update conversation history
                update_conversation_history(
                    user_id,
                    "user",
                    f"Uploaded PDF: {file_name}",
                    {"type": "pdf", "file_id": vector_store_file_id}
                )
                
                # Generate response about the PDF
                prompt = f"A new PDF document has been uploaded with ID: {vector_store_file_id}. Please search this document and provide a summary."
                response = await generate_text_with_agent(
                    prompt,
                    user_id,
                    {"type": "pdf", "file_id": vector_store_file_id}
                )
                
                # Add document reference
                doc_reference = format_document_reference(vector_store_file_id, file_name)
                full_response = f"{response}\n\n{doc_reference}"
                
                # Update assistant response in history
                update_conversation_history(user_id, "assistant", full_response)
                
                # Send response
                await send_message(user_id, full_response)
            else:
                await send_error(user_id, "vector_store")
        except Exception as e:
            print(f"Error processing PDF: {e}")
            await send_error(user_id, "pdf_upload")
    else:
        # Handle non-PDF files
        await send_error(user_id, "pdf_unsupported")

async def process_line_event(event: MessageEvent):
    """Process LINE event using a dispatcher pattern."""
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
        await send_error(user_id, "unsupported_message_type")
        return
        
    # Execute the handler
    try:
        await handler(event)
    except Exception as e:
        print(f"Error in {event.message.type} handler: {e}")
        await send_error(user_id, "general")

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

    # Process events in background tasks
    for event in events:
        if isinstance(event, MessageEvent):
            background_tasks.add_task(process_line_event, event)
    
    # Return 200 OK immediately
    return {"message": "OK"}

# Add cleanup for session on app shutdown
@app.on_event("shutdown")
async def cleanup():
    """Clean up resources on application shutdown."""
    await session.close()