import streamlit as st
from vector_store import load_and_chunk_documents, build_vector_store
from retriever import retrieve
from local_llm import ask_llm
from agent import route_query
from llama_cpp import Llama  # Assuming you are using the GGUF model for Hugging Face
from huggingface_hub import hf_hub_download


# Initialize system
st.title("🧠 RAG-Powered Multi-Agent Q&A Assistant")

# Load the model only once using hf_hub_download to get the model from Hugging Face
model_path = hf_hub_download(
    repo_id="Vamsidhara/RAG-ASSISTANT",  # Your model's repo on Hugging Face
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # The model file
)

# Initialize the model
llm = Llama(model_path=model_path, n_ctx=2048)

with st.spinner("Loading documents and building vector store..."):
    # Load and chunk documents, then build the vector store
    chunks = load_and_chunk_documents()
    index, _, all_chunks, model = build_vector_store(chunks)

query = st.text_input("Ask a question:")

if query:
    decision = route_query(query)
    st.write(f"Agent decision: `{decision}`")

    if decision == "RAG":
        # Retrieve top chunks from the vector store
        top_chunks = retrieve(query, index, model, all_chunks)
        context = "\n".join(top_chunks)
        prompt = f"Answer this based on the context:\n\n{context}\n\nQuestion: {query}"

        # Ask the LLM model for the response
        response = ask_llm(prompt, llm)  # Passing the `llm` model explicitly to the `ask_llm` function
        st.markdown("### Retrieved Context")
        st.code(context)
        st.markdown("### Answer")
        st.success(response)

    elif decision == "CALC":
        try:
            expression = query.lower().split("calculate")[-1].strip()
            result = eval(expression)  # Be cautious with eval, consider using a safer alternative
            st.success(f"Result: {result}")
        except:
            st.error("Couldn't calculate the expression.")

    elif decision == "DICT":
        word = query.lower().split("define")[-1].strip()
        st.info(f"Definition for '{word}': [Insert dictionary logic or use local glossary]")
