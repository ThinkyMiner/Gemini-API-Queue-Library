import os
from collections import deque
from dotenv import load_dotenv
import uuid
from google.ai import generativelanguage_v1beta as genai_services
from google.api_core import client_options
from . import persistence
from .context import BaseContextStrategy, SimpleContextStrategy, RollingSummaryStrategy

class GeminiManager:
    # __init__, _get_next_key, and context management methods are mostly unchanged
    def __init__(self, context_strategy: BaseContextStrategy = SimpleContextStrategy()):
        load_dotenv()
        api_keys_str = os.getenv("GEMINI_API_KEYS")
        if not api_keys_str: raise ValueError("GEMINI_API_KEYS not found in .env file.")
        self.api_keys = deque([key.strip() for key in api_keys_str.split(',')])
        self.context_strategy = context_strategy
        persistence._ensure_dir()
        print(f"GeminiManager initialized with {len(self.api_keys)} keys and '{type(context_strategy).__name__}' strategy.")
    def _get_next_key(self):
        current_key = self.api_keys[0]
        self.api_keys.rotate(-1)
        return current_key
    def list_contexts(self): return persistence.list_contexts()
    def create_context(self, context_id: str):
        initial_data = self.context_strategy.get_initial_state()
        persistence.create_new_context(context_id, initial_data)
    def delete_context(self, context_id: str): persistence.delete_context(context_id)


    # --- NEW ARCHITECTURE METHODS ---

    def get_client(self):
        """Rotates the API key and returns a configured client."""
        api_key = self._get_next_key()
        print(f"--- Providing client with API Key ending in: ...{api_key[-4:]} ---")
        opts = client_options.ClientOptions(api_key=api_key)
        return genai_services.GenerativeServiceClient(client_options=opts)

    def prepare_contents(self, prompt: str, context_id: str):
        """Loads, manages, and returns the 'contents' list for an API call."""
        if not persistence.context_exists(context_id):
            raise FileNotFoundError(
                f"Context '{context_id}' not found. "
                f"Please create it first using `manager.create_context('{context_id}')`."
            )

        context_data = persistence.load_context(context_id)
        
        # Pass a client to the strategy if it needs one (for summarization)
        client_for_summary = None
        if isinstance(self.context_strategy, RollingSummaryStrategy):
            client_for_summary = self.get_client()

        # The strategy prepares the history and updates the context data in-place (e.g., creates a new summary)
        history = self.context_strategy.prepare_history(context_data, client_for_summary)
        
        # Save any changes made during preparation (like a new summary)
        persistence.save_context(context_id, context_data)

        # Append the current prompt to the prepared history to form the final contents
        final_contents = history + [{"role": "user", "parts": [{"text": prompt}]}]
        return final_contents

    def update_context(self, prompt: str, response_text: str, context_id: str):
        """Saves the latest turn of conversation back to the context file."""
        if not persistence.context_exists(context_id):
            return # Should not happen in normal flow
        
        context_data = persistence.load_context(context_id)
        
        # Delegate the update logic to the strategy
        self.context_strategy.update_state(prompt, response_text, context_data)
        
        persistence.save_context(context_id, context_data)
        print(f"--- Context '{context_id}' updated. ---")