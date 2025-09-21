from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import tempfile
import os
import numpy as np
from typing import List, Dict, Any
import uvicorn
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query_embedding, k=4):
        if not self.embeddings:
            return []
        
        query_embedding = np.array(query_embedding)
        doc_embeddings = np.array(self.embeddings)
        
        dot_products = np.dot(doc_embeddings, query_embedding)
        norms = np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = dot_products / norms
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]

def process_json_document(file_path):
    """Process a JSON document for embedding."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        text_content = json.dumps(data, indent=2)
        
        doc = Document(page_content=text_content, metadata={"source": file_path})
        return [doc]
    except Exception as e:
        logger.error(f"Error processing JSON document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing JSON document: {str(e)}")

def process_pdf_document(file_path):
    """Process a PDF document for embedding."""
    try:
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]
        return docs
    except Exception as e:
        logger.error(f"Error processing PDF document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF document: {str(e)}")

def process_questions(file_path):
    """Process questions from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        elif isinstance(data, list):
            return data
        else:
            logger.error("Invalid questions format")
            raise HTTPException(status_code=400, detail="Invalid questions format. Expected a list of questions or a JSON object with a 'questions' key.")
    except Exception as e:
        logger.error(f"Error processing questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing questions: {str(e)}")

def get_embeddings_direct(texts):
    """Get embeddings directly using the OpenAI API without any libraries."""
    try:
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-ada-002",
                    "input": batch
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from OpenAI API: {response.text}")
            
            data = response.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

def chat_completion_direct(messages):
    """Get chat completion directly using the OpenAI API without any libraries."""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error getting chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chat completion: {str(e)}")

def create_qa_system(documents):
    """Create a question-answering system with the documents."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        from langchain_core.documents import Document as LCDocument
        lc_docs = [LCDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]
        
        split_docs = text_splitter.split_documents(lc_docs)
        
        splits = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in split_docs]
        
        texts = [doc.page_content for doc in splits]
        
        embeddings = get_embeddings_direct(texts)
        
        vector_store = SimpleVectorStore()
        vector_store.add_documents(splits, embeddings)
        
        def qa_function(question):
            query_embedding = get_embeddings_direct([question])[0]
            
            relevant_docs = vector_store.similarity_search(query_embedding, k=4)
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided documents."
            }
            
            user_message = {
                "role": "user",
                "content": f"""
                Context information is below:
                {context}
                
                Given the context information and not prior knowledge, answer the following question:
                {question}
                
                If the answer is not in the provided context, say "I don't have enough information to answer this question."
                """
            }
            
            answer = chat_completion_direct([system_message, user_message])
            
            return answer
        
        return qa_function
    except Exception as e:
        logger.error(f"Error creating QA system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating QA system: {str(e)}")

@app.post("/qa")
async def question_answering(
    document: UploadFile = File(...),
    questions_file: UploadFile = File(...),
):
    """
    Process a document and answer questions about it.
    
    - document: PDF or JSON file containing the content to be queried
    - questions_file: JSON file containing questions
    """
    document_path = None
    questions_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(document.filename)[1]) as temp_document:
            temp_document.write(await document.read())
            document_path = temp_document.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_questions:
            temp_questions.write(await questions_file.read())
            questions_path = temp_questions.name
        
        document_extension = os.path.splitext(document.filename)[1].lower()
        if document_extension == '.pdf':
            documents = process_pdf_document(document_path)
        elif document_extension == '.json':
            documents = process_json_document(document_path)
        else:
            os.unlink(document_path)
            os.unlink(questions_path)
            raise HTTPException(status_code=400, detail="Unsupported document format. Only PDF and JSON are supported.")
        
        questions = process_questions(questions_path)
        
        qa_system = create_qa_system(documents)
        
        results = []
        for q in questions:
            question_text = q.get('question', q) if isinstance(q, dict) else q
            answer = qa_system(question_text)
            
            if isinstance(q, dict):
                results.append({"id": q.get("id"), "question": question_text, "answer": answer})
            else:
                results.append({"question": question_text, "answer": answer})
        
        if document_path:
            os.unlink(document_path)
        if questions_path:
            os.unlink(questions_path)
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error in question_answering: {e}", exc_info=True)
        if document_path and os.path.exists(document_path):
            os.unlink(document_path)
        if questions_path and os.path.exists(questions_path):
            os.unlink(questions_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)