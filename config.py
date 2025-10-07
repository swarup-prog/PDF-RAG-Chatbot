# config.py
import os

# General config
CHROMA_PATH = "./chroma_db"
MEMORY_FILE = "memory.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
K = 3  # number of relevant chunks

# Ollama configuration
os.environ["OLLAMA_USE_GPU"] = "1" # Use GPU if available
