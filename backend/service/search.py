from qdrant_client import QdrantClient
from config import vector_db_path
from gigachat import GigaChat
from gigachat.models import Chat, Messages
from dotenv import load_dotenv
from sentence_transformers import (SentenceTransformer,
                                   CrossEncoder)
import os
import numpy as np
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
llm = GigaChat(credentials=os.getenv("GIGA_CHAT_API_KEY"),
               verify_ssl_certs=False)

model = SentenceTransformer("intfloat/multilingual-e5-base")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def search_documents(query_text: str, limit=2):
    query_vector = model.encode(query_text)
    # client = QdrantClient(path=vector_db_path)
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_KEY'),
        timeout=120)
    query_points = client.query_points(collection_name="help",
                                       query=query_vector,
                                       limit=20)
    points = [point.payload.get('text') for point in query_points.points]
    pairs = [(query_text, point.payload.get('text')) for point in query_points.points]
    scores = reranker.predict(pairs)
    indexes = np.argsort(scores)[::-1][:5]
    final_points = [points[index] for index in indexes]

    context = '\n'.join(point for point in final_points)

    """
    context = '\n\n'.join(f"Релевантность:{point.score}\n"
        f"{point.payload.get('text')}" for point in query_points.points)
    """

    system_instruction = f"""
        Ты отвечаешь на вопросы только по предоставленному CONTEXT.

        СТРОГИЕ ПРАВИЛА:

        1. Используй исключительно информацию из CONTEXT.
        2. Не используй свои знания о Fidesys, физике, инженерии или любых других темах.
        3. Не добавляй факты, которых буквально нет в CONTEXT.
        4. Не делай выводов, требующих внешних знаний.
        5. Если CONTEXT содержит только часть информации, ответь только этой частью.
        6. Если CONTEXT вообще не содержит ответа на вопрос, напиши:
           информация не найдена
        7. Не добавляй "информация не найдена", если хотя бы часть вопроса
           может быть достоверно отвечена по CONTEXT.
        8. Не копируй служебные теги [TITLE], [LIST], [PARAGRAPH] и т.д.

        CONTEXT:
        {context}
        """

    system_message = Messages(role='system', content=system_instruction)
    user_message = Messages(role='user', content=query_text)
    messages = [system_message, user_message]
    chat = Chat(messages=messages)
    response = llm.chat(chat)
    return {"query": response.choices[0].message.content,
            "limit": limit}
