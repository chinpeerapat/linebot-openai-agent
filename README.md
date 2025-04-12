# LINE Bot with OpenAI Responses API

## Project Background

This project is a LINE bot that uses OpenAI's Responses API to generate conversational responses to text inputs, images, and documents. The bot can answer questions in Thai and English, process images, and handle PDF documents with advanced search capabilities.

## Screenshot

![image](https://github.com/user-attachments/assets/61066eef-2802-4967-a5eb-e2a4e430e5f7)

## Features

- Multilingual support with Thai and English
- Text message processing using OpenAI's latest Responses API
- Web search capabilities for real-time information
- Image analysis with configurable detail levels
- PDF document processing with vector search capabilities
- Integration with LINE Messaging API for easy mobile access
- Built with FastAPI for efficient asynchronous processing

## Technologies Used

- Python 3
- FastAPI
- LINE Messaging API
- OpenAI Responses API
- Web search and file search tools
- Aiohttp
- PIL (Python Imaging Library)
- Vector Store for document search
- python-dotenv for environment configuration

## Setup

1. Clone the repository to your local machine.

2. Create a `.env` file by copying the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file with your configuration:
   ```env
   # OpenAI Configuration
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o

   # LINE Bot Configuration
   CHANNEL_SECRET=your_line_channel_secret_here
   CHANNEL_ACCESS_TOKEN=your_line_channel_access_token_here

   # Optional Vector Store Configuration
   VECTOR_STORE_ID=your_vector_store_id_here  # Optional

   # Feature Configuration
   WEB_SEARCH_CONTEXT_SIZE=low  # Options: low, medium, high
   IMAGE_DETAIL_LEVEL=high      # Options: low, high, auto
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

6. Set up your LINE bot webhook URL to point to your server's endpoint.

## Usage

### Text Processing

Send any text message to the LINE bot, and it will use the OpenAI Responses API to generate a response in the same language as your query (Thai or English). The bot can search the web for up-to-date information when needed.

### Image Analysis

Send an image to the bot, and it will provide a detailed analysis in the same language as your previous conversation. Image analysis detail level can be configured via environment variables.

### PDF Document Processing

Upload PDF files (up to 10MB) to the bot for:
- Automatic content analysis
- Vector store indexing for advanced search
- Bilingual document references
- Context-aware document querying

## Tool Capabilities

The bot uses several advanced tools from the OpenAI Responses API:

1. **Web Search**: Searches the internet for real-time information to provide up-to-date answers
2. **File Search**: Searches through uploaded PDF documents using vector embeddings
3. **Vision**: Analyzes and describes images with configurable detail levels

## Deployment Options

### Local Development

Use ngrok or similar tools to expose your local server to the internet for webhook access:

```
ngrok http 8000
```

### Docker Deployment

You can use the included Dockerfile to build and deploy the application:

```
docker build -t linebot-openai-agent .
docker run -p 8000:8000 \
  -e CHANNEL_SECRET=YOUR_SECRET \
  -e CHANNEL_ACCESS_TOKEN=YOUR_TOKEN \
  -e OPENAI_API_BASE=YOUR_BASE_URL \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  -e OPENAI_MODEL=YOUR_MODEL \
  -e VECTOR_STORE_ID=YOUR_VECTOR_STORE_ID \
  linebot-openai-agent
```

### Cloud Deployment

1. Install the Google Cloud SDK and authenticate with your Google Cloud account.
2. Build the Docker image:

   ```
   gcloud builds submit --tag gcr.io/$GOOGLE_PROJECT_ID/linebot-openai-agent
   ```

3. Deploy the Docker image to Cloud Run:

   ```
   gcloud run deploy linebot-openai-agent \
     --image gcr.io/$GOOGLE_PROJECT_ID/linebot-openai-agent \
     --platform managed \
     --region $REGION \
     --allow-unauthenticated
   ```

4. Set up your LINE bot webhook URL to point to the Cloud Run service URL.
