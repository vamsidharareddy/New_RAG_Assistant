from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)
output = llm("What is the capital of India?", max_tokens=50)
print(output['choices'][0]['text'])
