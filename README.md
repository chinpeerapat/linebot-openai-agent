# LINE Bot with OpenAI Agent

## Project Background

This project is a LINE bot that uses OpenAI Agent functionality to generate responses to text inputs. The bot can answer questions in Thai and English, process images, and handle PDF documents with advanced search capabilities.

## Screenshot

![image](https://github.com/user-attachments/assets/61066eef-2802-4967-a5eb-e2a4e430e5f7)


## Features

- Multilingual support with Thai and English
- Text message processing using OpenAI Agent
- Image analysis with scientific detail
- PDF document processing with vector search capabilities
- Integration with LINE Messaging API for easy mobile access
- Built with FastAPI for efficient asynchronous processing

## Technologies Used

- Python 3
- FastAPI
- LINE Messaging API
- OpenAI Agent framework
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
   EXAMPLE_BASE_URL=https://api.openai.com/v1
   EXAMPLE_API_KEY=your_openai_api_key_here
   EXAMPLE_MODEL_NAME=gpt-4

   # LINE Bot Configuration
   ChannelSecret=your_line_channel_secret_here
   ChannelAccessToken=your_line_channel_access_token_here

   # Optional Vector Store Configuration
   VECTOR_STORE_ID=your_vector_store_id_here  # Optional
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

Send any text message to the LINE bot, and it will use the OpenAI Agent to generate a response in the same language as your query (Thai or English).

### Image Analysis

Send an image to the bot, and it will provide a detailed scientific analysis in the same language as your previous conversation.

### PDF Document Processing

Upload PDF files (up to 10MB) to the bot for:
- Automatic content analysis
- Vector store indexing for advanced search
- Bilingual document references
- Context-aware document querying

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
  -e ChannelSecret=YOUR_SECRET \
  -e ChannelAccessToken=YOUR_TOKEN \
  -e EXAMPLE_BASE_URL=YOUR_BASE_URL \
  -e EXAMPLE_API_KEY=YOUR_API_KEY \
  -e EXAMPLE_MODEL_NAME=YOUR_MODEL \
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
