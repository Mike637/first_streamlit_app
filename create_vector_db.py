import os
from typing import List

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_DIR = os.path.dirname(__file__)
HELP_PATH = os.path.join(PROJECT_DIR, 'help')

'''
# Функция для поиска HTML файлов
def add_html_paths(folder_path) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        html_files = [f for f in files if f.endswith('.html') or f.endswith('.htm')]
        for html_file in html_files:
            html_path = os.path.join(path, html_file)
            if os.path.exists(html_path):
                with open(html_path, encoding='utf-8') as file_path:
                    file = file_path.read()
                html_files_list.append(html_path)
    return html_files_list
'''


def add_html_paths(folder_path) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.html', '.htm')):
                html_files_list.append(os.path.join(path, f))
    return html_files_list


html_paths = add_html_paths(HELP_PATH)

'''
class MyBSHTMLLoader:
    def __init__(self, file_path, parser='html.parser'):
        self.file_path = file_path
        self.parser = parser

    def load(self):
        with open(self.file_path, encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, self.parser)
        text = '\n'.join([line.strip() for line in soup.get_text('\n').splitlines() if line.strip()])
        return text
'''


def parse_html(path) -> str:
    try:
        with open(path, encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # text = '\n'.join([line.strip() for line in soup.get_text('\n').splitlines() if line.strip()])
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = '\n'.join([line.strip() for line in soup.get_text('\n').splitlines() if line.strip()])
        return text
    except Exception as e:
        print(f"Ошибка в файле {path}:{e}")
        return ""


def load_documents(paths):
    for path in paths:
        text = parse_html(path)
        if text:
            yield Document(page_content=parse_html(path), metadata={"source": os.path.basename(path)})


documents = list(load_documents(html_paths))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

chunks = text_splitter.split_documents(documents)

embed_texts = HuggingFaceEmbeddings(
    model_name='intfloat/multilingual-e5-large',
    encode_kwargs={'batch_size': 64}
)

vectore_store = QdrantVectorStore.from_documents(
    chunks,
    embed_texts,
    path="./qdrant_data",
    collection_name="my_collection",
    force_recreate=False)

print('База успешно сохранена')

