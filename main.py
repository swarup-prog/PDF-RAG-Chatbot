from vectorstore import vectorstore_init
from llm import generate_answer
from memory import load_memory, save_memory
from config import K

def chat():
  vector_store = vectorstore_init()
  memory = load_memory()

  print("\nRAG Chatbot ready! Type 'exit' to quit. \n")

  while True:
    query = input("You: "). strip()
    if query.lower() in ["exit", "quit"]:
      print("Goodbye!")
      save_memory(memory)
      break

    # Retrive revalant chunks
    results = vector_store.similarity_search(query, k=K)
    context_docs = "\n\n".join([doc.page_content for doc in results])

    # Use recent conversation as memory
    memory_context = "\n".join(
      [f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-3:]]
    )

    prompt = f"""
    You are a concise and factual assistant.
    Use only the provided document (primary) and chat history (secondary).
    Do not repeat or explain how you work.
    If unsure or off-topic, say: "The document doesn’t mention that."

    Be direct and polite.

    Chat:
    {memory_context}

    Document:
    {context_docs}

    Question: {query}
    Answer:
    """



    answer = generate_answer(prompt, query)
    memory.append({"user": query, "bot": answer})
    save_memory(memory)

    print(f"\nAssistant: {answer}\n")
    

if __name__ == "__main__":
  chat()



