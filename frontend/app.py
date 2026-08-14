import streamlit as st
from prompt_creation import ask

if __name__ == '__main__':
    st.title('My first App')
    query = st.text_input("Input your question")
    if query:
        st.text(ask(query))
