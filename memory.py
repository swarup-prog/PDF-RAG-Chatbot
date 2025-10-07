import json
import os
from config import MEMORY_FILE


def load_memory():
    """Load previous memory from JSON file or return an empty list if invalid."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if not data:
                    return []  # empty file
                return json.loads(data)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Memory file corrupted or empty — resetting memory.")
            return []
    return []


def save_memory(memory):
    """Save conversation history to JSON file."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
