import os
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                      HTMLHeaderTextSplitter,
                                      HTMLSectionSplitter
                                      )
from concurrent.futures import ThreadPoolExecutor
from typing import (Iterable,
                    Union)
from config import settings
from langchain_openai import OpenAIEmbeddings
import logging
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from tqdm import tqdm
import numpy as np

# MAX_WORKERS = os.cpu_count()
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 100
PROJECT_DIR = os.path.dirname(__file__)
#HELP_PATH = os.path.join(PROJECT_DIR, 'training_help', 'finite_element_analysis', 'automechanics')
HELP_PATH = os.path.join(PROJECT_DIR, 'help')
COLLECTION_NAME = "my_collection"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 128
EMBED_MODEL = "text-embedding-3-large"
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


def parse_html(path: str):
    html = Path(path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')

    # 1. удалить мусор RoboHelp
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()

    for tag in soup.select('#header, .WebHelpPopupMenu, .WebHelpNavBar'):
        tag.decompose()

    # 2. удалить пустые p
    for p in soup.find_all('p'):
        if not p.get_text(strip=True):
            p.decompose()

    # 3. картинки → текст (но НЕ удаляем структуру)
    for img in soup.find_all('img'):
        src = img.get("src", "")
        img.replace_with(f"[IMAGE: {src}]")

    # 4. код (без .string!)
    for code in soup.find_all(['pre', 'code']):
        code.replace_with(f"\nCODE_BLOCK:\n{code.get_text()}\n")
    return soup


def load_html_documents(folder_path) -> Iterable[str]:
    html_files_list = get_html_paths(folder_path)
    for path in html_files_list:
        html = parse_html(path)
        yield html


def split_htmls(folder_path) -> Iterable[Document]:
    headers_to_split_on = [
        ("h1", "section"),
        ("h2", "subsection"),
        ("h3", "subsubsection"),
        ("h4", "level4"),
        ("h5", "level5"),
        ("h6", "level6"),
    ]
    html_splitter = HTMLSectionSplitter(headers_to_split_on)
    htmls = load_html_documents(folder_path)
    for html in htmls:
        text = str(html)
        yield from html_splitter.split_text(text)


def clean_chunk_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')

    # таблицы → текст
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cols = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            rows.append(" | ".join(cols))
        table.replace_with("\nTABLE:\n" + "\n".join(rows) + "\n")

    # заголовки → текст (теперь можно!)
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text(strip=True)
        tag.replace_with(f"\nSection_{tag.name.upper()}: {text}\n")

    return soup.get_text("\n", strip=True)


def split_htmls_1(folder_path) -> Iterable[Document]:
    splitter = HTMLSectionSplitter([
        ("h1", "section"),
        ("h2", "subsection"),
        ("h3", "subsubsection"),
    ])

    for html in load_html_documents(folder_path):
        for doc in splitter.split_text(str(html)):
            doc.page_content = clean_chunk_text(doc.page_content)
            yield doc


def init_qdrant() -> QdrantVectorStore:
    client = QdrantClient(path="./qdrant_data")
    if RECREATE_COLLECTION:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            logger.warning("Удаляем старую коллекцию")

    if not client.collection_exists(COLLECTION_NAME):
        logger.info("Создаем коллекцию")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE
            )
        )
    embed_texts = OpenAIEmbeddings(
        api_key=settings.OPEN_AI_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model=EMBED_MODEL
    )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embed_texts
    )


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


html_path = get_html_paths(HELP_PATH)
print(os.path.exists(HELP_PATH))


# parse = parse_html(html_path[300])
# print(parse)


def main() -> None:
    html_paths = get_html_paths(HELP_PATH)
    if not html_paths:
        logger.warning('html файлы не найдены')
        return

    logger.info(f"Найдено {len(html_paths)}")
    vector_store = init_qdrant()
    chunks = split_htmls_1(HELP_PATH)
    index_batches(chunks, vector_store)
    logger.info("Индексация завершена")


if __name__ == '__main__':
    main()
