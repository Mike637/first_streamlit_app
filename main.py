import asyncio
import streamlit as st
import os
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from gigachat import GigaChat

key = 'YOUR_KEY_HERE'
st.title('My first App')

embed_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

vectorestore = QdrantVectorStore.from_existing_collection(
    path="./qdrant_data",
    collection_name="my_collection",
    embedding=embed_model
)
retriever = vectorestore.as_retriever(search_kwargs={"k": 6})

async def get_promt(query_text):
    results = retriever.get_relevant_documents(query_text)
    async with GigaChat(credentials=key, verify_ssl_certs=False) as giga:
        context = "\n".join([doc.page_content for doc in results])
        prompt = f"""
Ты эксперт по документации Fidesys.
Отвечай на основе переданного контекста.
Не используй внешние знания.
Если информации нет — отвечай: В документации нет информации.
Документы:
{context}

Вопрос: {query_text}

Ответ:
"""
        response = await giga.chat(prompt)
        return response.choices[0].message.content


with st.form(key='my_form'):
    question = st.text_input("Введите вопрос")
    submitted = st.form_submit_button("Спросить")
    if submitted:
        # запускаем async функцию через asyncio.run()
        answer = asyncio.run(get_promt(question))
        st.write(answer)