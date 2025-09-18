import streamlit as st
import requests

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

FASTAPI_URL = "http://localhost:8069"

st.title("🤖 RAG Enhanced Chatbot")
st.caption("Chat with your documents!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = requests.post(
                f"{FASTAPI_URL}/ask",
                json={"query": prompt},
                timeout=None  
            )
            response.raise_for_status() 
            
            api_response = response.json()
            full_response = api_response.get("answer", "No answer received from the backend.")
            
        except requests.exceptions.RequestException as e:
            full_response = f"An error occurred: {e}"
            
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})