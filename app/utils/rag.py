"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import os
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Config

import os
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

class Config:
    TRANSFORMER_EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    PDF_FOLDER = "./pdfs"  # путь к папке с PDF

def load_pdf(file_path: str) -> list:
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_pdfs(pdf_folder: str) -> list:
    docs = []

    pdf_files = [os.path.join(pdf_folder, file)
                 for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_files))

    for result in results:
        docs.extend(result)

    return docs

def build_index():
    embedding_model = HuggingFaceEmbeddings(model_name=Config.TRANSFORMER_EMBEDDINGS_MODEL_NAME)
    documents = load_pdfs(Config.PDF_FOLDER)
    texts = [doc.page_content for doc in documents]
    embeddings = np.array([embedding_model.embed_query(text) for text in texts], dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return embedding_model, index, texts

def search_docs(query: str, embedding_model, index, texts, k=3) -> list:
    query_vector = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_vector, k)
    return [texts[i] for i in indices[0]]

if __name__ == "__main__":
    embedding_model, index, texts = build_index()

    query = "What is the document about?"
    results = search_docs(query, embedding_model, index, texts)

    for i, res in enumerate(results, 1):
        print(f"Result {i}:\n{res}\n")


