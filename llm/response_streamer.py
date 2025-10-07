import sys, time
from typing import Iterable

def stream(chunks: Iterable[str], typing_delay: float = 0.01):
  """Streams response to terminal with typing effect."""
  for text in chunks:
    sys.stdout.write(text)
    sys.stdout.flush()
    time.sleep(typing_delay)
