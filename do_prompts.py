from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import os
from langchain_core.prompts import PromptTemplate
from create_vector_db import vectorestore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from gigachat import GigaChat
from config import gigachat_key

retriever = vectorestore.as_retriever(search_kwargs={"k": 3})
query = "типы конечных элементов в фидесисе"
results = retriever.get_relevant_documents(query)

with GigaChat(credentials=gigachat_key, verify_ssl_certs=False) as giga:
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    Ты — эксперт, который может отвечать только на основе предоставленных документов.
    Используй **только** эти документы. Не добавляй информации, которой нет в них.
    Если не знаешь , ответь я не знаю
    Документы:
    {context}

    Вопрос: {query}

    Ответ:
    """

    response = giga.chat(prompt)
    print("Ответ GigaChat:", response.choices[0].message.content)