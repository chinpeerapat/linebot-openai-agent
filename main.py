import os
import sys
import asyncio
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
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
if not all([BASE_URL, API_KEY, MODEL_NAME, CHANNEL_SECRET, CHANNEL_ACCESS_TOKEN]):
    missing_vars = [
        var_name for var_name, var_value in {
            "EXAMPLE_BASE_URL": BASE_URL,
            "EXAMPLE_API_KEY": API_KEY,
            "EXAMPLE_MODEL_NAME": MODEL_NAME,
            "ChannelSecret": CHANNEL_SECRET,
            "ChannelAccessToken": CHANNEL_ACCESS_TOKEN
        }.items() if not var_value
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI()

# Initialize LINE bot API client
http_client = AiohttpAsyncHttpClient()
parser = WebhookParser(CHANNEL_SECRET)
line_bot_api = AsyncLineBotApi(CHANNEL_ACCESS_TOKEN, http_client)

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Initialize conversation histories
conversation_histories: Dict[str, List[Dict[str, Any]]] = {}

# Image processing prompt
image_prompt = '''
Please describe this image with scientific detail.
Provide a detailed analysis focusing on scientific aspects and notable features.
Respond in the same language as the user's query.
'''

# File processing prompts
pdf_prompt = '''
Analyze this PDF document and provide a detailed summary.
Focus on key points, main ideas, tables, and figures.
Respond in the same language as the user's query.
'''

# Maximum conversation history length
MAX_HISTORY_LENGTH = 10

# Error messages
ERROR_MESSAGES = {
    "image_processing": "ไม่สามารถประมวลผลรูปภาพได้ โปรดตรวจสอบว่าเป็นไฟล์ JPEG หรือ PNG\nUnable to process the image. Please ensure it's in a supported format (JPEG, PNG).",
    "pdf_upload": "ไม่สามารถอัปโหลด PDF ได้ โปรดตรวจสอบว่าเป็นไฟล์ PDF ที่ถูกต้อง\nUnable to upload the PDF to our system. Please ensure it's a valid PDF file.",
    "pdf_unsupported": "รองรับเฉพาะไฟล์ PDF เท่านั้น\nThis file type is not supported. Currently, we only support PDF files.",
    "vector_store": "ได้รับเอกสารแล้วแต่ไม่สามารถจัดเก็บสำหรับการค้นหาขั้นสูงได้\nUnable to store the document for future reference. The document was received but cannot be used for advanced queries.",
    "general": "เกิดข้อผิดพลาดที่ไม่คาดคิด โปรดลองอีกครั้งในภายหลัง\nAn unexpected error occurred. Please try again later.",
    "file_too_large": "ไฟล์มีขนาดใหญ่เกินไป โปรดอัปโหลดไฟล์ที่มีขนาดไม่เกิน 10MB\nFile is too large. Please upload a file smaller than 10MB.",
    "invalid_format": "รูปแบบไฟล์ไม่ถูกต้อง โปรดตรวจสอบว่าไฟล์ไม่เสียหาย\nInvalid file format. Please ensure the file is not corrupted."
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

# Initialize the agent with proper tools
agent = Agent(
    name="LINE Bot Assistant",
    instructions="""You are a helpful assistant that can communicate in both Thai and English.
    - Respond in the same language as the user's query
    - For Thai users, provide responses in Thai with English translations when appropriate
    - For English users, respond in English
    - You can search the web for real-time information
    - You can search through stored documents when given a Document ID
    - For PDFs:
        - When users mention a Document ID, use the file_search tool to retrieve relevant information
        - Provide document summaries and answer specific questions about the content
    - Be concise but informative in your responses
    """,
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=[VECTOR_STORE_ID] if VECTOR_STORE_ID else [],
            include_search_results=True,
        )
    ],
)

async def process_image(message_id: str) -> str:
    """
    Process an image message and return its content.
    """
    try:
        message_content = await line_bot_api.get_message_content(message_id)
        image_data = BytesIO()
        async for chunk in message_content:
            image_data.write(chunk)
        image_data.seek(0)
        
        # Convert to PIL Image for potential preprocessing
        image = PIL.Image.open(image_data)
        
        # Convert image to base64 for OpenAI vision model
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            }
        }
        
        return image_content
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def update_conversation_history(user_id: str, role: str, content: str, image_content: Optional[dict] = None):
    """
    Update the conversation history for a specific user.
    """
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    
    message = {"role": role, "content": content}
    if image_content:
        message["image"] = image_content
    
    conversation_histories[user_id].append(message)
    
    # Keep only the last MAX_HISTORY_LENGTH messages
    if len(conversation_histories[user_id]) > MAX_HISTORY_LENGTH:
        conversation_histories[user_id] = conversation_histories[user_id][-MAX_HISTORY_LENGTH:]

async def format_error_message(error_type: str, details: Optional[str] = None, language: str = "en") -> str:
    """
    Format error messages based on error type and user's language.
    Includes both Thai and English for better user experience.
    """
    base_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general"])
    if details and not any(x in str(details).lower() for x in ["http", "error", "exception", "traceback"]):
        # Only include safe, user-friendly details
        safe_details = ' '.join(str(details).split()[:10])  # First 10 words
        base_message = f"{base_message}\n\nข้อมูลเพิ่มเติม | Additional info: {safe_details}"
    return base_message

def format_document_reference(doc_id: str, doc_name: str) -> str:
    """
    Format document reference information in a consistent way with both Thai and English.
    """
    # Clean and truncate document name if too long
    safe_doc_name = doc_name[:50] + "..." if len(doc_name) > 50 else doc_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return DOCUMENT_REFERENCE.format(
        doc_id=doc_id,
        doc_name=safe_doc_name,
        timestamp=timestamp
    )

async def process_pdf(message_id: str, file_name: str) -> Tuple[str, Optional[str]]:
    """
    Process a PDF file message and upload to vector store.
    Returns a tuple of (status_message, vector_store_file_id).
    """
    try:
        message_content = await line_bot_api.get_message_content(message_id)
        pdf_data = BytesIO()
        total_size = 0
        async for chunk in message_content:
            total_size += len(chunk)
            if total_size > 10 * 1024 * 1024:  # 10MB limit
                error_msg = await format_error_message("file_too_large")
                return error_msg, None
            pdf_data.write(chunk)
        pdf_data.seek(0)
        
        try:
            # Upload to OpenAI's vector store
            file_upload = await client.files.create(
                file=pdf_data,
                purpose="assistants",
                file_name=file_name
            )
            vector_store_file_id = file_upload.id
            print(f"Successfully uploaded PDF to vector store with ID: {vector_store_file_id}")
            return "Success", vector_store_file_id
            
        except Exception as e:
            print(f"Error uploading to vector store: {e}")
            error_msg = await format_error_message("vector_store", str(e))
            return error_msg, None
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        error_msg = await format_error_message("pdf_upload", str(e))
        raise HTTPException(status_code=400, detail=error_msg)

@app.post("/")
async def handle_callback(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Process events in background tasks
    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        
        # Create background task for processing
        background_tasks.add_task(process_line_event, event)
    
    # Return 200 OK immediately
    return {"message": "OK"}

async def process_line_event(event: MessageEvent):
    """
    Process LINE event in background to avoid timeout.
    """
    try:
        user_id = event.source.user_id
        print(f"Processing message from user: {user_id}")

        if event.message.type == "text":
            # Process text message
            msg = event.message.text
            print(f"Received text message: {msg}")

            response = await generate_text_with_agent(msg, user_id)
            reply_msg = TextSendMessage(text=response)
            await line_bot_api.push_message(event.source.user_id, reply_msg)
            
        elif event.message.type == "image":
            print("Received image message")
            try:
                # Process the image
                image_content = await process_image(event.message.id)
                
                # Generate response using the image
                response = await generate_text_with_agent(image_prompt, user_id, image_content)
                reply_msg = TextSendMessage(text=response)
                await line_bot_api.push_message(event.source.user_id, reply_msg)
                
            except Exception as e:
                print(f"Error processing image message: {e}")
                error_msg = await format_error_message("image_processing")
                await line_bot_api.push_message(
                    event.source.user_id,
                    TextSendMessage(text=error_msg)
                )
        elif event.message.type == "file":
            print("Received file message")
            try:
                # Check file size first
                if event.message.file_size > 10 * 1024 * 1024:  # 10MB limit
                    error_msg = await format_error_message("file_too_large")
                    await line_bot_api.push_message(
                        event.source.user_id,
                        TextSendMessage(text=error_msg)
                    )
                    return

                # Check if it's a PDF file
                if event.message.file_name.lower().endswith('.pdf'):
                    print("Processing PDF file")
                    # Process the PDF and get both status and vector store ID
                    status_message, vector_store_file_id = await process_pdf(event.message.id, event.message.file_name)
                    
                    if vector_store_file_id:
                        # Update conversation history with PDF context
                        update_conversation_history(
                            user_id,
                            "user",
                            f"Uploaded PDF: {event.message.file_name}",
                            {"type": "pdf", "file_id": vector_store_file_id}
                        )
                        
                        # Generate response about the PDF using file search
                        response = await generate_text_with_agent(
                            f"A new PDF document has been uploaded with ID: {vector_store_file_id}. Please search this document and provide a summary.",
                            user_id,
                            {
                                "type": "pdf",
                                "file_id": vector_store_file_id
                            }
                        )
                        
                        # Add formatted document reference to the response
                        doc_reference = format_document_reference(
                            vector_store_file_id,
                            event.message.file_name
                        )
                        response = f"{response}\n\n{doc_reference}"
                    else:
                        response = await format_error_message("vector_store", status_message)
                    
                    await line_bot_api.push_message(
                        event.source.user_id,
                        TextSendMessage(text=response)
                    )
                else:
                    # Handle non-PDF files
                    error_msg = await format_error_message("pdf_unsupported")
                    await line_bot_api.push_message(
                        event.source.user_id,
                        TextSendMessage(text=error_msg)
                    )
            except Exception as e:
                print(f"Error processing file message: {e}")
                error_msg = await format_error_message("general", str(e))
                await line_bot_api.push_message(
                    event.source.user_id,
                    TextSendMessage(text=error_msg)
                )
    except Exception as e:
        print(f"Error in background task: {e}")
        try:
            error_msg = await format_error_message("general", str(e))
            await line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=error_msg)
            )
        except:
            print(f"Failed to send error message to user: {e}")

async def generate_text_with_agent(prompt: str, user_id: str, content: Optional[dict] = None):
    """
    Generate text response using the agent with conversation history and context.
    """
    try:
        # Get conversation history
        conversation_context = ""
        if user_id in conversation_histories:
            history = conversation_histories[user_id]
            conversation_context = "Previous conversation:\n" + "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in history
            )

        # Add PDF context if available
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

        # Update conversation history
        update_conversation_history(user_id, "user", prompt)
        update_conversation_history(user_id, "assistant", response)

        return response

    except Exception as e:
        print(f"Error generating text: {e}")
        error_msg = await format_error_message("general", str(e))
        return error_msg
