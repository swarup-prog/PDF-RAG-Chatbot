import subprocess
from utils import count_tokens


def generate_answer(prompt: str) -> str:
  input_tokens = count_tokens(prompt)

  result = subprocess.run(
    ["ollama", "run", "llama3"],
      input=prompt,
      text=True,
      capture_output=True,
      encoding="utf-8",
      errors="ignore"
  )

  output = result.stdout.strip()
  output_tokens = count_tokens(output)
  total_tokens = input_tokens + output_tokens

  print(f"\nToken usage => Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}\n")
  return output
