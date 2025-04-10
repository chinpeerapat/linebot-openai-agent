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
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled, WebSearchTool, FileSearchTool

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
BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or "vs_67f79305f05481919ea528bb2df2ade3"

# LINE Bot configuration
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)

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

# Validate environment variables
if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Initialize OpenAI client
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# Initialize conversation history storage
conversation_histories: Dict[str, List[dict]] = {}

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
    Generate a text completion using OpenAI Agent with conversation history.
    """
    # Create agent with appropriate instructions
    tools = [WebSearchTool()]
    
    # Add FileSearchTool if vector store ID is configured
    if VECTOR_STORE_ID:
        file_search = FileSearchTool(
            max_num_results=3,
            vector_store_ids=[VECTOR_STORE_ID],
        )
        tools.append(file_search)
    
    # Prepare conversation context
    conversation_context = ""
    if user_id in conversation_histories:
        conversation_context = "Previous conversation:\n"
        for msg in conversation_histories[user_id]:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            msg_content = msg["content"]
            if "image" in msg:
                msg_content = "📷 [Image shared] " + msg_content
            elif "pdf" in msg:
                msg_content = "📄 [PDF shared] " + msg_content
                if msg.get("file_id"):
                    msg_content += f" [Doc ID: {msg['file_id']}]"
            conversation_context += f"{prefix}{msg_content}\n"
    
    # Prepare the full prompt with context
    full_prompt = f"{conversation_context}\nCurrent request: {prompt}"
    if content:
        if content.get("type") == "image":
            full_prompt = f"{conversation_context}\nCurrent request: {prompt}\n[Image analysis requested]"
        elif content.get("type") == "pdf":
            pdf_text = content.get("content", "")
            file_id = content.get("file_id")
            if file_id:
                full_prompt = f"{conversation_context}\nCurrent request: {prompt}\nPDF Content (Document ID: {file_id}):\n{pdf_text}"
            else:
                full_prompt = f"{conversation_context}\nCurrent request: {prompt}\nPDF Content:\n{pdf_text}"
    
    agent = Agent(
        name="Assistant",
        instructions="""You are a helpful assistant that can communicate in multiple languages.
        You should detect the language of the user's query and respond in the same language.
       
        
        You can understand text, images, and PDF documents, and provide informative responses.
        You can search the web for real-time information and through stored documents when needed.
        Maintain conversation context and refer to previous messages when relevant.
        
        For images:
        - Provide detailed scientific analysis
        - Respond in the same language as the user's query or previous conversation
        
        For PDFs:
        - Analyze the content and provide comprehensive summaries
        - Maintain the same language as the user's query
        - When users mention a Document ID, use the file_search tool to retrieve relevant information
        
        Always maintain the conversation in the user's preferred language throughout the interaction.""",
        model=OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=client),
        tools=tools,
    )

    try:
        # Run the agent with the full context
        result = await Runner.run(agent, full_prompt)
        response = result.final_output
        
        # Update conversation history
        update_conversation_history(user_id, "user", prompt, content)
        update_conversation_history(user_id, "assistant", response)
        
        return response
    except Exception as e:
        print(f"Error with OpenAI Agent: {e}")
        error_msg = await format_error_message("general", str(e))
        update_conversation_history(user_id, "system", error_msg)
        return error_msg
