def build_prompt(query: str, memory_context: str, docs_context: str = "") -> str:
    return f"""
    You are a helpful assistant that answers based on the provided document and remembers past interactions.
    - Be direct and concise.
    - Use memory and document context for answers.
    - If unrelated, say you can only answer based on the document.

    Conversation:
    {memory_context}

    Document Context:
    {docs_context}

    User Question: {query}

    Assistant:
    """
