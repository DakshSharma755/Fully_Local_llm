import streamlit as st
import requests
import time

FASTAPI_URL = "http://localhost:8069" 

st.title("📄 Document Upload")
st.caption("Upload one or more PDF documents to add to the chatbot's knowledge base.")

uploaded_files = st.file_uploader(
    "Click to select one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.write("Selected Documents:")
    for file in uploaded_files:
        st.write(f"- {file.name}")

    if st.button("Add to Knowledge Base"):
        with st.spinner("Processing and uploading files... This may take a while."):
            all_successful = True
            for file in uploaded_files:
                try:
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    
                    response = requests.post(
                        f"{FASTAPI_URL}/upload-document/",
                        files=files,
                        timeout=300
                    )
                    response.raise_for_status()
                    
                    time.sleep(1)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error uploading {file.name}: {e}")
                    all_successful = False

            if all_successful:
                st.success("All selected documents have been successfully added to the knowledge base!")