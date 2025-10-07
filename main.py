from rag import vectorstore
from llm import generate_answer, build_prompt
from memory import load_memory, save_memory
from config import K

def chat():
  vector_store = vectorstore()
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
    docs_context = "\n\n".join([doc.page_content for doc in results])

    # Use recent conversation as memory
    memory_context = "\n".join(
      [f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-3:]]
    )

    prompt = build_prompt(query, memory_context, docs_context)

    answer = generate_answer(prompt, query)
    memory.append({"user": query, "bot": answer})
    save_memory(memory)

    print(f"\nAssistant: {answer}\n")
    

if __name__ == "__main__":
  chat()



