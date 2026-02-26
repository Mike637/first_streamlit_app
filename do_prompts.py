from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import os
from langchain_core.prompts import PromptTemplate
from create_vector_db import vectorestore
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM

'''
DIR_NAME = os.path.dirname(__file__)
index = faiss.read_index(os.path.join(DIR_NAME, 'my_faiss_index', 'index.faiss'))
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
query_text = 'APREPRO'
query_vector = model.encode(query_text, convert_to_numpy=True).astype('float32')
distances, indices = index.search(query_vector.reshape(1, -1), k=20)
'''
'''
for index in indices[0]:
    print(texts[index])
    
    print('______________________________________')
'''
'''

new_texts = list(map(lambda el: texts[el], indices[0]))
print(new_texts)
'''
retriever = vectorestore.as_retriever()
template = """
Используй контекст, чтобы ответить на вопрос. 
Можешь использовать свои знания на русском языке
Отвечай только по сути, без вступительных пояснений и английских фраз.
Если не знаешь что ответить, пиши - Я не знаю
Дай ** 3 разных варианта ответа** в виде списка на языке програмирования python
Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

custom_rag_prompt = PromptTemplate.from_template(template)
llm = ChatOllama(model="llama3")
#llm = AutoModelForCausalLM.from_pretrained("NousResearch/Nous-Hermes-13b", device_map="auto")
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
)
print(rag_chain.invoke("Какой алгоритм работы KNN"))
