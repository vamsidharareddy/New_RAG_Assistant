# vector_store.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

def load_and_chunk_documents(folder="docs/"):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            text = f.read()
            chunks.extend(splitter.split_text(text))
    
    return chunks

def build_vector_store(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings, chunks, model
