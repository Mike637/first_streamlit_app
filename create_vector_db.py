import os
from typing import List
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_DIR = os.path.dirname(__file__)
HELP_PATH = os.path.join(PROJECT_DIR, 'help')


def add_html_paths(folder_path) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.html', '.htm')):
                html_files_list.append(os.path.join(path, f))
    return html_files_list


def parse_html(path) -> str:
    try:
        with open(path, encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        for tag in soup(['nav', 'script', 'style', 'footer', 'header']):
            tag.decompose()
        text = '\n'.join(line.strip() for line in soup.get_text('\n').splitlines() if line.strip())
        return text
    except Exception as e:
        print(f"Ошибка в файле {path}:{e}")
        return ""


def iter_chunks(documents, splitter):
    for doc in documents:
        for chunk in splitter.split_documents([doc]):
            yield chunk


def indexes_in_batches(chunks, vector_db, batch_size=128):
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            vector_db.add_documents(batch)
            batch.clear()
    if batch:
        vector_db.add_documents(batch)


def thread_iter_documents(html_paths):
    max_workers = min(32, os.cpu_count() * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        contents = executor.map(parse_html, html_paths)
        for path, content in zip(html_paths, contents):
            if content:
                yield Document(page_content=content, metadata={"source": os.path.basename(path)})


def main():
    html_paths = add_html_paths(HELP_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embed_texts = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-large',
        encode_kwargs={'batch_size': 64,
                       'normalize_embeddings': True}
    )
    client = QdrantClient(path="./qdrant_data")
    if client.collection_exists("my_collection"):
        client.delete_collection("my_collection")

    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_collection",
        embedding=embed_texts
    )
    documents = thread_iter_documents(html_paths)
    chunks = iter_chunks(documents, text_splitter)
    indexes_in_batches(chunks, vector_store)


if __name__ == '__main__':
    main()
    print('База успешно сохранена')
