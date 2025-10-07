import redis
import json
from hashlib import sha256
from config import REDIS_HOST, REDIS_PORT, REDIS_DB
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
import numpy as np

class RedisSemanticCache:
  def __init__(self, threshold=0.9):
    self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    self.threshold = threshold

  def _cosine_similarity(self, vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  
  def find_similar(self, query):
        """Find semantically similar cached query."""
        query_emb = self.embedder.embed_query(query)
        all_keys = self.client.keys("semcache:*") 

        for key in all_keys:
            data = self.client.hgetall(key)
            cached_emb = np.frombuffer(data[b'emb'], dtype=np.float32)
            sim = self._cosine_similarity(query_emb, cached_emb)
            if sim >= self.threshold:
                return data[b'answer'].decode("utf-8")
  
  def store(self, query, answer, expiry_seconds=86400):
    """
    Store a query-answer pair in Redis with optional expiry.
    
    Args:
        query (str): User query text.
        answer (str): Model-generated answer.
        expiry_seconds (int): Time in seconds before cache expires (default: 1 day).
    """
    emb = np.array(self.embedder.embed_query(query), dtype=np.float32)
    key = f"semcache:{sha256(query.encode()).hexdigest()}"

    # Store embedding + answer in hash
    self.client.hset(key, mapping={
        "answer": answer,
        "emb": emb.tobytes()
    })

    # Set expiry (TTL) on this key
    self.client.expire(key, expiry_seconds)