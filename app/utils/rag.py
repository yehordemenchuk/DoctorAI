import os
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Config

def load_pdf(file_path: str) -> list:
    loader = PyPDFLoader(file_path)

    return loader.load()

def load_pdfs(pdf_folder: str) -> list:
    docs = []

    pdf_files = [os.path.join(pdf_folder, file)
                 for file in os.listdir(pdf_folder) if file.endswith(".pdf")]


    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_pdf, pdf_files))

    for result in results:
        docs.extend(result)

    return docs

embedding_model = HuggingFaceEmbeddings(model_name=Config.TRANSFORMER_EMBEDDINGS_MODEL_NAME)

documents = load_pdfs(Config.PDF_FOLDER)

texts = [doc.page_content for doc in documents]

embeddings = np.array([embedding_model.embed_query(text) for text in texts], dtype=np.float32)

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

def search_docs(query: str, k=3) -> list:
    query_vector = np.array([embedding_model.embed_query(query)])

    distances, indices = index.search(query_vector, k)

    return [texts[i] for i in indices[0]]

