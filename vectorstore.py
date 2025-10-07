import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

document_path = "./data/langchain-research.pdf"
doc_name = document_path.split("/")[-1].replace(".pdf", "")
chroma_subdir = os.path.join(CHROMA_PATH, doc_name)

def vectorstore_init():
  embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

  if os.path.exists(chroma_subdir):
    # Load existing vectorstore
    vector_store = Chroma(persist_directory=chroma_subdir, embedding_function=embedding)
    print("Loaded existing vector store.")
  else:
    print("Creating new vector store...")
    loader = PyPDFLoader(document_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
      chunk_size=CHUNK_SIZE,
      chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Create new vectorstore
    vector_store = Chroma.from_documents(chunks, embedding, persist_directory=chroma_subdir)
    print("Data loaded in ChromaDB")
  
  return vector_store






