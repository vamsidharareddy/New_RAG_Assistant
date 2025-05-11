# retriever.py
def retrieve(query, index, model, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]
