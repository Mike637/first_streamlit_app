from qdrant_client import QdrantClient
from config import vector_db_path
from gigachat import GigaChat
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

load_dotenv()
llm = GigaChat(credentials=os.getenv("GIGA_CHAT_API_KEY"))
query_text = "Что такое фидесис?"
model = SentenceTransformer("intfloat/multilingual-e5-base")
query_vector = model.encode(query_text)
client = QdrantClient(path=vector_db_path)
query_points = client.query_points(collection_name="help",
                                     query=query_vector,
                                     limit=5)
context = '\n'.join(point.payload.get('text') for point in query_points.points )
print(context)
