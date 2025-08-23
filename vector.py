from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Variables
DB_LOCATION = "./chrome_langchain_db"
ADD_DOCUMENTS = not os.path.exists(DB_LOCATION)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
K_VECTOR = 10
SCORE_TRESHOLD_VECTOR = 0.4

#Load embedding model
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

#Add different manuals
pdf_files = ["Documents/live12-manual-en.pdf",
            "Documents/101_Ableton_Tips.pdf",
            "Documents/MakingMusic_DennisDeSantis.pdf",
            "Documents/Ableton12.pdf"
            #"Documents/.pdf", #Use when adding more pdf's
]

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE, 
    chunk_overlap = CHUNK_OVERLAP
)

docs = [] #Where the documents end up

if ADD_DOCUMENTS:
#Split into chunks
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
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

if ADD_DOCUMENTS:
    vector_store.add_documents(docs)
    print("New documents added and saved")
else:
    print("Using existing database.")

#creates retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": K_VECTOR,
                   "score_threshold": SCORE_TRESHOLD_VECTOR
                   }
)
    
print("PDF embeddngs are in:", DB_LOCATION)