from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_community.llms import LlamaCpp
# Embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# Vector store
vectorstore = QdrantVectorStore.from_existing_collection(
    path="./qdrant_data",
    collection_name="my_collection",
    embedding=embed_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM
llm = LlamaCpp(
    model_path="./model/mistral-7b-instruct-v0.2.Q4_0.gguf",
    n_ctx = 8192,
    n_threads = 8
)

# Prompt template
template = """
Ты эксперт по документации Fidesys.

Правила:
- Отвечай ТОЛЬКО на основе предоставленных документов.
- Не используй внешние знания.
- Если в документах нет ответа — напиши: "Ответ не найден в документации".
- Не придумывай информацию.

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
query = "что такое ортотропия"

# retrieve docs
docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

# format prompt
final_prompt = prompt.format(context=context, question=query)

# LLM answer
response = llm.invoke(final_prompt,max_tokens = 1024)

print(response)
#print(model.generate(prompt))

