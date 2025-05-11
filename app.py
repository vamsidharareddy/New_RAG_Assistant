import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.title("🧠 RAG-Powered Multi-Agent Q&A Assistant")

# Step 1: Start message
st.write("👋 App started successfully!")

# Step 2: Download model
st.write("⏳ Downloading GGUF model from Hugging Face...")
try:
    model_path = hf_hub_download(
        repo_id="vamsidhara/RAG-ASSISTANT",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    )
    st.success("✅ Model downloaded.")
except Exception as e:
    st.error(f"❌ Model download failed: {e}")
    st.stop()

# Step 3: Load model with llama-cpp
st.write("⏳ Loading model with llama-cpp...")
try:
    llm = Llama(model_path=model_path, n_ctx=2048)
    st.success("✅ Model loaded.")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Step 4: Import and process data
st.write("📚 Loading and chunking documents...")
try:
    from vector_store import load_and_chunk_documents, build_vector_store
    from retriever import retrieve
    from local_llm import ask_llm
    from agent import route_query

    chunks = load_and_chunk_documents()
    index, _, all_chunks, model = build_vector_store(chunks)
    st.success("✅ Vector store built.")
except Exception as e:
    st.error(f"❌ Error building vector store: {e}")
    st.stop()

# Step 5: Main user input and processing
query = st.text_input("Ask a question:")

if query:
    decision = route_query(query)
    st.write(f"Agent decision: `{decision}`")

    if decision == "RAG":
        try:
            top_chunks = retrieve(query, index, model, all_chunks)
            context = "\n".join(top_chunks)
            prompt = f"Answer this based on the context:\n\n{context}\n\nQuestion: {query}"
            response = ask_llm(prompt, llm)
            st.markdown("### Retrieved Context")
            st.code(context)
            st.markdown("### Answer")
            st.success(response)
        except Exception as e:
            st.error(f"❌ Error during RAG pipeline: {e}")

    elif decision == "CALC":
        try:
            expression = query.lower().split("calculate")[-1].strip()
            result = eval(expression)
            st.success(f"Result: {result}")
        except Exception as e:
            st.error(f"❌ Calculation error: {e}")

    elif decision == "DICT":
        word = query.lower().split("define")[-1].strip()
        st.info(f"Definition for '{word}': [Insert dictionary logic or glossary here]")