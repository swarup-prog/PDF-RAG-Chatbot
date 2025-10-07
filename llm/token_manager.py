from utils import count_tokens

def get_token_usage(prompt: str, output: str):
  input_tokens = count_tokens(prompt)
  output_tokens = count_tokens(output)
  total = input_tokens + output_tokens
  print(f"\n\nToken usage => Input: {input_tokens} | Output: {output_tokens} | Total: {total}\n")
  return input_tokens, output_tokens, total
