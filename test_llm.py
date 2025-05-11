from llama_cpp import Llama

llm = Llama(model_path="models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf", n_ctx=2048)
output = llm("What is the capital of India?", max_tokens=50)
print(output['choices'][0]['text'])
