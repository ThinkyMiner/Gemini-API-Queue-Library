import os
import json
from typing import List, Dict, Any

CONTEXTS_DIR = "contexts"

def _ensure_dir() -> None:
    """Ensures the context directory exists."""
    os.makedirs(CONTEXTS_DIR, exist_ok=True)

def _get_path(context_id: str) -> str:
    """Gets the full path for a given context ID."""
    return os.path.join(CONTEXTS_DIR, f"{context_id}.json")

def context_exists(context_id: str) -> bool:
    """Checks if a context file already exists."""
    return os.path.exists(_get_path(context_id))

def create_new_context(context_id: str, initial_data: Any) -> None:
    """Creates a new context file, raising an error if it already exists."""
    if context_exists(context_id):
        raise FileExistsError(f"Context '{context_id}' already exists.")
    save_context(context_id, initial_data)
    print(f"Successfully created context '{context_id}'.")

def load_context(context_id: str) -> Any:
    """Loads context data from a JSON file."""
    if not context_exists(context_id):
        raise FileNotFoundError(f"Context '{context_id}' not found. Create it first with `create_context`.")
    with open(_get_path(context_id), 'r') as f:
        return json.load(f)

def save_context(context_id: str, data: Any) -> None:
    """Saves context data to a JSON file."""
    _ensure_dir()
    with open(_get_path(context_id), 'w') as f:
        json.dump(data, f, indent=2)

def list_contexts() -> List[str]:
    """Lists all available context IDs."""
    _ensure_dir()
    return [f.replace(".json", "") for f in os.listdir(CONTEXTS_DIR) if f.endswith(".json")]

def delete_context(context_id: str) -> None:
    """Deletes a context file."""
    if context_exists(context_id):
        os.remove(_get_path(context_id))
        print(f"Successfully deleted context '{context_id}'.")
    else:
        print(f"Context '{context_id}' not found, nothing to delete.")