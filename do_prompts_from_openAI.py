from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from config import settings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings

LLM = ChatOpenAI(
    api_key=settings.OPEN_AI_KEY.get_secret_value(),
    base_url=settings.BASE_URL,
    model='gpt-5.4-mini',
    timeout=30
)

# embs = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

embs = OpenAIEmbeddings(
    api_key=settings.OPEN_AI_KEY.get_secret_value(),
    base_url=settings.BASE_URL,
    model="text-embedding-3-large"
)

vectorstore = QdrantVectorStore.from_existing_collection(
    path="./qdrant_data",
    collection_name="my_collection",
    embedding=embs,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8,
                   "fetch_k": 40,
                   "lambda_mult": 0.5}
)

template = """
Ты отвечаешь строго по документации.
Ответ должен быть в обычном тексте без Markdown-разметки
Правила:
1. Используй только информацию из документов
2. Разрешено переформулировать и обобщать
3. Если есть несколько фрагментов:
   - выбери тот, который наиболее полно отвечает на вопрос
   - приоритет имеют перечисления и явные исключения
4. Формулировки вида:
   - "исключение составляют"
   - "не включается"
   - "не записывается"
   считать прямым ответом на вопрос
5. Используй ВСЮ релевантную информацию, а не первое совпадение
6. Если релевантной информации нет → напиши:
"Ответ не найден в документации"

Дай развернутый ответ.

Документы:
{context}

Вопрос:
{question}

Ответ:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# user query

query = "Какая особенность есть в оссеметричной задаче?"

docs = retriever.get_relevant_documents(query)
for i, doc in enumerate(docs):
    print(doc.page_content)
    print('________________________________')
qa_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    retriever=retriever,
    # chain_type="refine",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
response = qa_chain.invoke({"query": query})
print(response['result'])
