from llama_cpp import Llama

# Load the model globally
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)

def ask_llm(prompt, llm=llm):
    """
    Given a prompt, run it through the provided LLM model and return the generated text.
    If no LLM is provided, it defaults to the globally loaded model.
    """
    print("🔧 Running LLM on prompt...")
    output = llm(prompt, max_tokens=256, stop=["</s>"])
    return output['choices'][0]['text'].strip()
