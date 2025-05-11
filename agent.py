# agent.py
def route_query(query):
    query = query.lower()
    if "calculate" in query:
        return "CALC"
    elif "define" in query:
        return "DICT"
    else:
        return "RAG"
