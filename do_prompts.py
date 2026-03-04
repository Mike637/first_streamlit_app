
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
query = "роторной динамика"
results = retriever.get_relevant_documents(query)
with GigaChat(credentials=gigachat_key, verify_ssl_certs=False, temperature = 0) as giga:
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"""
   "Ты эксперт по документации Fidesys. "
            "Анализируй документы и формируй ответ строго на их основе. "
            "делай логические выводы, но только из представленного контекста. "
            "Не добавляй внешние знания. "
            "Выводи только итоговый ответ без пояснений."
    Документы:
    {context}

    Вопрос: {query}

    Ответ:
    """
    response = giga.chat(prompt)
    print("Ответ GigaChat:", response.choices[0].message.content)

