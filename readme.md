# Local RAG Chatbot

This project is a complete, self-contained Retrieval-Augmented Generation (RAG) chatbot that runs entirely on local hardware. It uses a quantized open-source Large Language Model (LLM) accelerated by a local GPU. The application can ingest a knowledge base of PDF documents, answer questions based on their content, and be updated with new documents on the fly through an interactive web interface.

The entire system is decoupled, featuring a robust FastAPI backend for the core logic and a user-friendly Streamlit frontend for interaction.



---
## Features

* **Advanced RAG Pipeline:** Implements a sophisticated two-stage retrieval process with initial vector search followed by a more accurate Cross-Encoder re-ranker.
* **Local, GPU-Accelerated LLM:** Runs a highly quantized GGUF model (e.g., `Phi-3.5-mini`) entirely on a local NVIDIA GPU, ensuring privacy and zero API costs.
* **Dynamic Knowledge Base:** Ingests and processes PDF documents, chunking them into a searchable vector index using LanceDB.
* **Live Updates:** Users can upload new PDF documents through the web UI, which are automatically indexed and added to the chatbot's knowledge base.
* **Interactive Web UI:** A simple, multi-page web application built with Streamlit, featuring a chat interface and a document management page.
* **Decoupled Architecture:** A robust FastAPI backend serves the AI/RAG logic, cleanly separated from the Streamlit frontend.

---
## Technology Stack

* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **LLM Engine:** `llama-cpp-python`
* **RAG Core:**
    * **Vector Store:** LanceDB
    * **Embedding & Re-ranking:** Sentence Transformers
    * **Text Processing:** LangChain (for text splitting), PyMuPDF (for PDF reading)
* **Core Language:** Python 3.12

---
## Setup & Installation

Follow these steps to set up the project environment on a WSL 2 (Ubuntu) instance.

### 1. Prerequisites: NVIDIA Driver & CUDA
Ensure you have a compatible NVIDIA driver installed on your Windows host. This project was built and tested using the **CUDA Toolkit 12.8** inside WSL.

* If needed, purge any existing CUDA installations and perform a clean install of the recommended toolkit version.
* **Crucially, set the environment variables** by adding the following lines to the end of your `~/.bashrc` file and then **restarting your terminal**:
    ```bash
    export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

### 2. Clone the Repository
```bash
git clone https://github.com/DakshSharma755/Fully_Local_llm
cd Fully_Local_llm
```

### 3. Set Up Python Environment

Create and activate a Python virtual environment.
```bash 
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Dependencies

Installation is a two-step process to ensure GPU acceleration.

First, build llama-cpp-python with CUDA support:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```
Then, install the rest of the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

### 5. Download a GGUF Model

Download a pre-quantized GGUF model and place it in the root of the project directory. This project is configured for a ~4B parameter model.

    Recommended Model: Phi-3.5-mini-Instruct-Q6_K.gguf

    Update the LOCAL_MODEL_PATH variable in llm_core.py to match the filename of your downloaded model.

### 6. Add Source Documents

Place the PDF files that will form the initial knowledge base into the /data directory. The application will automatically process them on its first run.

---
### Running the Application

This project requires two separate terminals to run the backend and frontend simultaneously

## Terminal 1: Start the Backend (FastAPI)

Navigate to the project directory, activate the environment, and run the Uvicorn server.

```bash
cd ~/rag_e_chatbot
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8069 --reload --log-config log_config.yaml
```
The backend will start, and on the first run, it will process all PDFs in the /data folder, create embeddings, and build the LanceDB vector store. This may take several minutes.

## Terminal 2: Start the Frontend (Streamlit)

Open a new terminal, navigate to the same project directory, and activate the same environment.
```bash
cd ~/rag_e_chatbot
source .venv/bin/activate
streamlit run frontend/chat.py
```
Streamlit will provide a URL (usually http://localhost:8501) to open in your web browser.

### How to Use

Once both servers are running, open the Streamlit URL in your browser.

    Chat Page: This is the main interface. Type your questions into the chat box at the bottom to get answers based on the indexed documents.

    Upload Documents Page: Use the sidebar navigation to go to the upload page. Here you can select and upload new PDF files to expand the chatbot's knowledge base in real-time.


### Project Structure

.
├──  .gitignore
├──  app.py
├──  llm_core.py
├──  log_config.yaml
├──  rag_core.py
├──  requirements.txt
├──  readme.md
├──  Phi-3.5-mini-Instruct-Q6_K.gguf
├──  data/
└──  frontend/
    ├──  chat.py              
    └──  pages/

        └──  uploaddocs.py
