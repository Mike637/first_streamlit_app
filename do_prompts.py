from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from gigachat import GigaChat
from config import gigachat_key
from langchain_qdrant import QdrantVectorStore

embed_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

vectorestore = QdrantVectorStore.from_existing_collection(
    path="./qdrant_data",
    collection_name="my_collection",
    embedding=embed_model
)

retriever = vectorestore.as_retriever(search_kwargs={"k": 6})
query = "как  импортировать файл cdb"
results = retriever.get_relevant_documents(query)
print(results)
with GigaChat(credentials=gigachat_key, verify_ssl_certs=False) as giga:
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    Ты эксперт по документации Fidesys.
Отвечай  на основе переданного контекста.
Не используй внешние знания.
Если информации нет — отвечай: В документации нет информации.
    Документы:
    {context}

    Вопрос: {query}

    Ответ:
    """

    response = giga.chat(prompt)
    print("Ответ GigaChat:", response.choices[0].message.content)

print(context)