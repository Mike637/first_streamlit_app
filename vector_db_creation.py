from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from chunks_creation import chunked_docs_by_tags
from sentence_transformers import SentenceTransformer
from config import vector_db_path

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
    client = QdrantClient(path=vector_db_path)
    client.create_collection(collection_name='help',
                             vectors_config=VectorParams(
                                 size=768,
                                 distance=Distance.COSINE
                             ))

    client.upsert(
        collection_name="help",
        points=points
    )
