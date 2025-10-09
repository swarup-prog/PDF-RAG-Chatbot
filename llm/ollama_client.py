import ollama
from cache import RedisSemanticCache
from .response_streamer import stream
from .token_manager import get_token_usage

cache = RedisSemanticCache()

def generate_answer(prompt: str, user_query: str) -> str:
    # check cache for similar query responses
    cached_response = cache.find_similar(user_query)
    if cached_response:
        print("Cache hit! Returning cached response.\n")
        return cached_response

    print("\nGenerating response...\n")
    output_chunks = []

    try:
        response_stream = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": """
                    You are a helpful assistant that answers based on the provided document and remembers past interactions.
                    - Be direct and concise.
                    - Use memory and document context for answers.
                    - If unrelated, say you can only answer based on the document.
                """},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in response_stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                stream(content)
                output_chunks.append(content)

    except Exception as e:
        print(f"\n[Error] Ollama streaming error: {e}")
        return "There was an error generating the response."

    output = "".join(output_chunks).strip()
    get_token_usage(prompt, output)
    cache.store(user_query, output)
    return output
