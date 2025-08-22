from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Load embedding model
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

#Add different manuals
pdf_files = ["Documents/live12-manual-en.pdf",
            "Documents/101_Ableton_Tips.pdf",
            "Documents/MakingMusic_DennisDeSantis.pdf",
            "Documents/Ableton12.pdf"
            #"Documents/.pdf", #Use when adding more pdf's
]

#database location
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
#Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        #pages = [page.extract_text() for page in reader.pages]
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks = splitter.split_text(text)
                docs.extend(
                    [Document(page_content=chunk, 
                              metadata={"source": os.path.basename(pdf), "page": page_number + 1}) 
                              for chunk in chunks])

vector_store = Chroma(
    collection_name="ableton_manual",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(docs)
    print("New documents added and saved")
else:
    print("Using existing database.")

#creates retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 100}
)
    
print("PDF embeddngs are in:", db_location)