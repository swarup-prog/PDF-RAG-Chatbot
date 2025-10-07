import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, EMBEDDING_MODEL
from .document_loader import load_and_split_pdf

document_path = "./data/langchain-research.pdf"
doc_name = document_path.split("/")[-1].replace(".pdf", "")
chroma_subdir = os.path.join(CHROMA_PATH, doc_name)

def vectorstore():
  embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

  if os.path.exists(chroma_subdir):
    # Load existing vectorstore
    vector_store = Chroma(persist_directory=chroma_subdir, embedding_function=embedding)
    print("Loaded existing vector store.")
  else:
    chunks = load_and_split_pdf(document_path)

    # Create new vectorstore
    vector_store = Chroma.from_documents(chunks, embedding, persist_directory=chroma_subdir)
    print("Data loaded in ChromaDB")
  
  return vector_store






