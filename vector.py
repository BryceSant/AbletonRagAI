from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location 