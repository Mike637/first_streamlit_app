import streamlit as st
import requests
from requests.exceptions import RequestException

st.title("Fastapi training")


query = st.text_input("input your question")

#limit = st.number_input('input number',min_value=1,
                        #max_value=20,
                        #value=5)

if query:
    try:
        response = requests.post("http://localhost:8000/search",
                                 json={
                                     "query": query,
                                     "limit":1
                                 })
        st.write(response.json().get("query"," "))

        response.raise_for_status()
    except RequestException as e:
        st.error(f"Error is {e}")
