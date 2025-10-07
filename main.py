import subprocess
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

    # Use convo memory
    # Use recent conversation as memory
    memory_context = "\n".join(
      [f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-3:]]
    )

    prompt= f"""
    You are a helpful assistant that answers based on the provided document which also have memory of past interactions.
    - Use the conversation and document context to answer.
    - Vary the response length and style depending on the question.
    - Be **direct and relevant** — avoid repeating information or giving long introductions.
    - Do **not** assume.
    - If the question is unrelated to the document, reply politely say that you can only answer based on the document.
    - Do **not** explain how you work.

    Conversation: 
    {memory_context}

    Document Context: 
    {context_docs}

    User Question: {query}

    Assistant:
    """

    answer = generate_answer(prompt)
    memory.append({"user": query, "bot": answer})
    save_memory(memory)

    print(f"\nAssistant: {answer}\n")
    

if __name__ == "__main__":
  chat()



