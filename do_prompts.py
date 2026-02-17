from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import os
from create_vector_db import texts

DIR_NAME = os.path.dirname(__file__)
index = faiss.read_index(os.path.join(DIR_NAME, 'my_faiss_index', 'index.faiss'))
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
query_text = 'APREPRO'
query_vector = model.encode(query_text,convert_to_numpy=True).astype('float32')
distances, indices = index.search(query_vector.reshape(1, -1), k=20)

for index in indices[0]:
    print(texts[index])
    print('______________________________________')