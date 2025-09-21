# Document Q&A API

A backend API that functions as a question-answering bot leveraging the capabilities of OpenAI's language models and the LangChain framework. The API answers questions based on the content of uploaded documents.

## Features

- Process both PDF and JSON documents
- Answer multiple questions in a single request
- Utilize LangChain and OpenAI's GPT-4o-mini for high-quality answers
- Store document embeddings in a FAISS vector database for efficient retrieval

## Prerequisites

- Python 3.8+
- OpenAI API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sowjanyachapala9/Zania-coding
cd ZANIA-QA-API
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
```
Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the API

Start the FastAPI server:
```bash
python app.py
```

The API will be available at http://localhost:8000

For API documentation, visit http://localhost:8000/docs

## API Endpoints

### POST /qa
Process a document and answer questions about it.

**Parameters:**
- `document_file`: PDF or JSON file containing the document content
- `questions_file`: JSON file with an array of questions
- `document_type`: "pdf" or "json"


## Architecture

- **FastAPI**: Handles HTTP requests and file uploads
- **LangChain**: Orchestrates the document processing and question answering workflow
- **OpenAI**: Provides the language model capabilities (GPT-4o-mini)
- **FAISS**: Acts as the vector database for storing document embeddings

## Security Considerations

- The API key is stored as an environment variable, not in the code
- Temporary files are properly cleaned up after processing

## Future Improvements

- Add authentication to the API
- Support more document formats (e.g., DOCX, TXT)
- Implement caching to improve performance for repeated questions
- Add more advanced document preprocessing options
