import os
from typing import List
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
'''
PROJECT_DIR = os.path.dirname(__file__)
HELP_PATH = os.path.join(PROJECT_DIR,'help') # замените на ваш путь

# Функция для поиска HTML файлов
def add_html_paths(folder_path) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        html_files = [f for f in files if f.endswith('.html') or f.endswith('.htm')]
        for html_file in html_files:
            html_path = os.path.join(path, html_file)
            if os.path.exists(html_path):
                html_files_list.append(html_path)
    return html_files_list

# Загружаем HTML файлы
html_paths = add_html_paths(HELP_PATH)


# Класс для извлечения текста из HTML
class MyBSHTMLLoader:
    def __init__(self, file_path, parser='html.parser'):
        self.file_path = file_path
        self.parser = parser

    def load(self):
        with open(self.file_path, encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, self.parser)
        return soup.get_text()

# Загружаем документы
documents = [Document(page_content=MyBSHTMLLoader(path).load()) for path in html_paths]

# Разделитель текста
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)

# Разделяем документы
all_splits = text_splitter.split_documents(documents)

# Получаем список текстов
texts = [doc.page_content for doc in all_splits]

# Инициализация модели для эмбеддингов
#model = SentenceTransformer("all-MiniLM-L6-v2")

# Функция для получения эмбеддингов
if __name__ == '__main__':
    embed_texts = HuggingFaceEmbeddings(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')

    # Создаем FAISS базу, передавая функцию эмбеддингов
    faiss_db = FAISS.from_texts(texts, embed_texts)

    # Сохраняем базу
    faiss_db.save_local("my_faiss_index")

    print('База успешно сохранена')
'''
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

loader = TextLoader("file.txt",encoding = 'utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=508,chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vectorestore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    path="./qdrant_data",
    collection_name="my_collection")