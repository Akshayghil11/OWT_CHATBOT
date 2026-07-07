import os
import threading
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "owt-new-project")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "d:/OWT"
MODEL_ID = str(
    Path(CACHE_DIR)
    / "models--Qwen--Qwen2.5-3B-Instruct"
    / "snapshots"
    / "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

# Global instances for reuse
_vectorstore = None
_llm = None
_llm_lock = threading.Lock()

def ensure_pinecone_index(pc, index_name):
    available_indexes = pc.list_indexes().names()
    if index_name not in available_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        print(f"Created Pinecone index '{index_name}'.")
    return index_name

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
            raise ValueError("Pinecone API Key is missing or invalid.")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = ensure_pinecone_index(pc, PINECONE_INDEX_NAME)
        print(f"DEBUG: Initializing PineconeVectorStore with index_name={index_name}")
        _vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name, 
            embedding=embeddings
        )
    return _vectorstore

def get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                print(f"Loading Qwen2.5-3B-Instruct from {MODEL_ID}...")
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
                
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    device_map="auto"
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=384,
                    max_length=2048,
                    temperature=0.5,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                _llm = HuggingFacePipeline(
                    pipeline=pipe,
                    model_kwargs={"max_new_tokens": 384, "max_length": 2048}
                )
    return _llm

import re

def is_greeting(text: str) -> bool:
    text = text.strip().lower()
    if text in {"greetings", "how are you", "what's up", "good morning", "good afternoon", "good evening"}:
        return True
    if re.match(r'^(hi+|hello+|hey+|hlo+|helo+)[a-z]*$', text):
        return True
    return False

def get_answer(question: str) -> str:
    # Handle greetings quickly to save response time
    if is_greeting(question):
        return "Hello! I am the OneWorld Technologies AI Assistant. How can I help you today?"

    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        
        # Retrieve context (reduced k to 3 to speed up response time)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt_template = """<|im_start|>system
You are an AI assistant for OneWorld Technologies.
Use the Context to answer the User's question.
If the question is unrelated to the Context or is nonsense, reply exactly: "I can only answer questions related to OneWorld Technologies based on my knowledge base."
Constraints:
1. Provide a clear, comprehensive, and complete answer. Do not cut off mid-sentence.
2. Do NOT start your response with "Answer:", "Our answer:", or any similar prefix.
3. Do NOT hallucinate or make up any information.<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=question)
        
        # Generate answer
        response = llm.invoke(final_prompt)
        
        # Post-process to remove prefixes
        response = response.strip()
        lower_resp = response.lower()
        for p in ["our answer:", "answer:"]:
            if lower_resp.startswith(p):
                response = response[len(p):].strip()
                break
                
        return response
        
    except Exception as e:
        print(f"Error during RAG: {e}")
        return f"An error occurred while generating the response. Ensure Pinecone credentials are set and the model is downloaded. Details: {str(e)}"

def get_answer_stream(question: str):
    # Handle greetings quickly to save response time
    if is_greeting(question):
        yield "Hello! I am the OneWorld Technologies AI Assistant. How can I help you today?"
        return

    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        
        # Retrieve context (reduced k to 3 to speed up response time)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt using ChatML format for Qwen
        prompt_template = """<|im_start|>system
You are an AI assistant for OneWorld Technologies.
Use the Context to answer the User's question.
If the question is unrelated to the Context or is nonsense, reply exactly: "I can only answer questions related to OneWorld Technologies based on my knowledge base."
Constraints:
1. Provide a clear, comprehensive, and complete answer. Do not cut off mid-sentence.
2. Do NOT start your response with "Answer:", "Our answer:", or any similar prefix.
3. Do NOT hallucinate or make up any information.<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=question)
        
        # Generate answer with programmatic stream filtering for prefixes only
        buffer = ""
        prefix_stripped = False
        unwanted_prefixes = ["our answer:", "answer:"]
        
        for chunk in llm.stream(final_prompt, pipeline_kwargs={"max_new_tokens": 384, "max_length": 2048}):
            if not prefix_stripped:
                buffer += chunk
                lower_buf = buffer.lstrip().lower()
                
                # Check if it might be building towards a prefix
                is_partial = False
                for p in unwanted_prefixes:
                    if p.startswith(lower_buf):
                        is_partial = True
                        break
                
                if is_partial:
                    continue  # Keep accumulating to see if it finishes the prefix
                
                # If not a partial match anymore, check if it actually started with it
                for p in unwanted_prefixes:
                    if lower_buf.startswith(p):
                        buffer = buffer.lstrip()[len(p):].lstrip()
                        break
                
                prefix_stripped = True
                yield buffer
            else:
                yield chunk
            
    except Exception as e:
        print(f"Error during RAG: {e}")
        yield f"An error occurred while generating the response. Details: {str(e)}"
