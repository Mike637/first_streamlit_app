from sentence_transformers import SentenceTransformer
import warnings
from chunks_creation import chunked_docs_by_tags



warnings.filterwarnings('ignore')
model = SentenceTransformer("intfloat/multilingual-e5-base")
texts = [chunk.page_content for chunk in chunked_docs_by_tags]
embeddings = model.encode(texts)
print(embeddings)