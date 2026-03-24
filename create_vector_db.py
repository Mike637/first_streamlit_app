import logging
import os
from typing import (List,
                    Iterable)
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_DIR = os.path.dirname(__file__)
HELP_PATH = os.path.join(PROJECT_DIR, 'help')

COLLECTION_NAME = "my_collection"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 128
EMBED_MODEL = "intfloat/multilingual-e5-large"
RECREATE_COLLECTION = True
MAX_WORKERS = os.cpu_count()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s "
)
logger = logging.getLogger(__name__)


def get_html_paths(folder_path: str) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.html', '.htm')):
                html_files_list.append(os.path.join(path, f))
    return html_files_list


def parse_html(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        for tag in soup([
            'nav', 'script', 'style', 'footer', 'header',
            'aside', 'noscript', 'form'
        ]):
            tag.decompose()
        text = '\n'.join(line.strip() for line in soup.get_text('\n').splitlines() if line.strip())
        return text
    except Exception as e:
        logger.error(f"Ошибка в файле {path}:{e}")
        return ""


def split_documents(documents: Iterable[Document],
                    splitter: RecursiveCharacterTextSplitter) -> Iterable[Document]:
    for doc in documents:
        yield from splitter.split_documents([doc])


def index_batches(chunks: Iterable[Document],
                  vector_db: QdrantVectorStore,
                  batch_size: int = BATCH_SIZE) -> None:
    batch = []
    for chunk in tqdm(chunks, desc="Indexing"):
        batch.append(chunk)
        if len(batch) >= batch_size:
            vector_db.add_documents(batch)
            batch.clear()
    if batch:
        vector_db.add_documents(batch)


def load_documents_parallel(html_paths: List[str]) -> Iterable[Document]:
    max_workers = min(32, MAX_WORKERS * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        contents = executor.map(parse_html, html_paths)
        for path, content in zip(html_paths, contents):
            if content:
                yield Document(page_content=content, metadata={
                    "source": path,
                    "filename": os.path.basename(path)
                })


def init_qdrant() -> QdrantVectorStore:
    client = QdrantClient(path="./qdrant_data")
    if RECREATE_COLLECTION:
        if client.collection_exists(COLLECTION_NAME):
            logger.warning("Удаляем старую коллекцию")

    if not client.collection_exists(COLLECTION_NAME):
        logger.info("Создаем коллекцию")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    embed_texts = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={'batch_size': 64,
                       'normalize_embeddings': True}
    )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embed_texts
    )


def main() -> None:
    html_paths = get_html_paths(HELP_PATH)
    if not html_paths:
        logger.warning('html файлы не найдены')
        return

    logger.info(f"Найдено {len(html_paths)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP
                                                   )
    vector_store = init_qdrant()
    documents = load_documents_parallel(html_paths)
    chunks = split_documents(documents, text_splitter)
    index_batches(chunks, vector_store)
    logger.info("Индексация завершена")


if __name__ == '__main__':
    main()
