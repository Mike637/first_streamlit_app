import streamlit as st
import requests
from requests.exceptions import RequestException

st.title("Fastapi training")
button = st.button('Нажми на кнопку')

if button:
    response = requests.get(
        "http://localhost:8000/health"
    )
    st.write(response.json())

query = st.text_input("input your question")
limit = st.number_input('input number',min_value=1,
                        max_value=20,
                        value=5)
if query:
    try:
        response = requests.post("http://localhost:8000/search",
                                 json={
                                     "query": query,
                                     "limit":limit
                                 })
        st.write(response.json())

        response.raise_for_status()
    except RequestException as e:
        st.error(f"Error is {e}")
