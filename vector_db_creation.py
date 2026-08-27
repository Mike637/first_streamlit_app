from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from chunks_creation import chunked_docs_by_tags
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from config import vector_db_path
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 10
model = SentenceTransformer("intfloat/multilingual-e5-base")
texts = [doc.page_content for doc in chunked_docs_by_tags]
vectors = model.encode(texts,
                       batch_size=128,
                       show_progress_bar=True)

points = [PointStruct(id=index,
                      vector=vector,
                      payload={
                          "text": doc.page_content,
                          **doc.metadata
                      })
          for index, (vector, doc) in enumerate(zip(vectors, chunked_docs_by_tags))]

if __name__ == '__main__':
    # client = QdrantClient(path=vector_db_path)
    # client = QdrantClient(url="http://localhost:6333")
    # print(client.get_collections())
    # client.delete_collection("help")
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_KEY'),
        timeout= 120
    )
    if not client.collection_exists('help'):
        client.create_collection(collection_name='help',
                                 vectors_config=VectorParams(
                                     size=768,
                                     distance=Distance.COSINE
                                 ))
    for i in tqdm(range(0, len(points), BATCH_SIZE)):
        client.upsert(
            collection_name="help",
            points=points[i:i + BATCH_SIZE]
        )
'''
print("QDRANT_URL:", os.getenv("QDRANT_URL"))
print("QDRANT_KEY exists:", os.getenv("QDRANT_KEY") is not None)
'''