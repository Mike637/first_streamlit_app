from qdrant_client import QdrantClient

def connect_collection():
    return QdrantClient(url="http://localhost:6333")