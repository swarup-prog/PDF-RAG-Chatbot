def build_prompt(query: str, memory_context: str, docs_context: str = "") -> str:
    return f"""
    Conversation:
    {memory_context}

    Document Context:
    {docs_context}

    User Question: {query}

    Assistant:
    """
