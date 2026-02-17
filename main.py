import streamlit as st
import os
import faiss
from sentence_transformers import SentenceTransformer
from create_vector_db import texts

DIR_NAME = os.path.dirname(__file__)
index = faiss.read_index(os.path.join(DIR_NAME, 'my_faiss_index', 'index.faiss'))
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
st.title('My first App')


def get_promt(query_text):
    query_vector = model.encode(query_text, convert_to_numpy=True).astype('float32')
    _, indices = index.search(query_vector.reshape(1, -1), k=20)
    new_texts = list(map(lambda el: texts[el], indices[0]))
    st.write(new_texts)



with st.form(key='my_form'):
    question = st.text_input("input question")
    submitted = st.form_submit_button("click")
    if submitted:
        get_promt(question)
