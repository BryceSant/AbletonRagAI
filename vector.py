from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

#Load embedding model
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

#Add Ableton Manual
pdf_path = "Documents/live12-manual-en.pdf"
reader = PdfReader(pdf_path)
pages = [page.extract_text() for page in reader.pages]

#database location
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)


if add_documents:
#Split into chunks
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = []
    for page in pages:
        if page:
            chunks = splitter.split_text(page)
            docs.extend([Document(page_content=chunk) for chunk in chunks])



vector_store = Chroma(
    collection_name="ableton_manual",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=docs)

print("PDF embeddngs are in:", db_location)

#creates retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 100}
)
    