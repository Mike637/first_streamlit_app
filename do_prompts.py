from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
import multiprocessing
# Embeddings

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
reranker = CrossEncoderReranker(
    model=model,
    top_n=5
)

embed_model = HuggingFaceEmbeddings(
    model_name='intfloat/multilingual-e5-large'
)

# Vector store
vectorstore = QdrantVectorStore.from_existing_collection(
    path="./qdrant_data",
    collection_name="my_collection",
    embedding=embed_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 50
    })

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=reranker
)
# LLM
llm = LlamaCpp(
    model_path="./model/mistral-7b-instruct-v0.2.Q4_0.gguf",
    verbose = False,
    n_ctx=10000,
    n_threads= multiprocessing.cpu_count(),
    temperature=0.1,
    model_kwargs = {"log_level":0},
    n_batch = 512
)

# Prompt template
template = """
Ты отвечаешь строго по документации.

Правила:
1. Используй ТОЛЬКО факты из документов
2. Если ответа нет → напиши:
"Ответ не найден в документации"
3. Запрещено:
   - придумывать
   - добавлять знания
   - делать выводы
   - продолжать диалог

Отвечай КРАТКО и ТОЧНО.

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
query = "что такое ортотропия ? "

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    #chain_type="refine",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
response = qa_chain.invoke({"query": query})
print(response['result'])

