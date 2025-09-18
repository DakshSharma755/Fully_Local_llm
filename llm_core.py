from llama_cpp import Llama

LOCAL_MODEL_PATH = "./Phi-3.5-mini-Instruct-Q6_K.gguf"

llm = None

def initialize_llm():
    """
    Loads the local GGUF model into memory.
    """
    global llm
    print(f"   - Loading local LLM from: {LOCAL_MODEL_PATH}...")
    llm = Llama(
        model_path=LOCAL_MODEL_PATH,
        n_gpu_layers=-99,  
        n_ctx=4096,  
        verbose=False
    )

def generate_answer(prompt: str):
    """
    Takes a fully constructed prompt and generates an answer using the local LLM.
    """
    if not llm:
        raise RuntimeError("LLM is not initialized. Call initialize_llm() first.")
        
    try:
        response = llm(
            prompt,
            max_tokens=1024,
            stop=["<|end|>"],
            echo=False
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error during local LLM call: {e}")
        return "Sorry, I encountered an error while generating a response."