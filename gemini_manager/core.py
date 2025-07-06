import os
from collections import deque
from dotenv import load_dotenv
import uuid
from google.ai import generativelanguage_v1beta as genai_services
from google.api_core import client_options
from . import persistence
from .context import BaseContextStrategy, SimpleContextStrategy, RollingSummaryStrategy, RetrievalAugmentedStrategy

class GeminiManager:
    # ... (all methods before prepare_contents are unchanged) ...
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
    def get_client(self):
        api_key = self._get_next_key()
        print(f"--- Providing client with API Key ending in: ...{api_key[-4:]} ---")
        opts = client_options.ClientOptions(api_key=api_key)
        return genai_services.GenerativeServiceClient(client_options=opts)


    def prepare_contents(self, prompt: str, context_id: str):
        if not persistence.context_exists(context_id):
            raise FileNotFoundError(f"Context '{context_id}' not found.")

        context_data = persistence.load_context(context_id)
        
        if isinstance(self.context_strategy, RetrievalAugmentedStrategy):
            rag_strategy = self.context_strategy
            collection_name = context_data['collection_name']
            
            query_vector = rag_strategy.embedding_model.encode(prompt).tolist()
            
            # --- THE FIX IS HERE ---
            # We add a `score_threshold` to filter out irrelevant results,
            # even if they fall within the top_k.
            search_results = rag_strategy.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=rag_strategy.top_k,
                score_threshold=0.6  # Only return results with a similarity of 0.6 or higher
            )
            
            if search_results:
                retrieved_context = "\n---\n".join([hit.payload['text'] for hit in search_results])
                augmented_prompt = (f"Given the following relevant information from our past conversation:\n--- BEGIN CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n\nNow, please answer this question: {prompt}")
            else:
                augmented_prompt = prompt
            
            return [{"role": "user", "parts": [{"text": augmented_prompt}]}]
        
        client_for_summary = self.get_client() if isinstance(self.context_strategy, RollingSummaryStrategy) else None
        history = self.context_strategy.prepare_history(context_data, client=client_for_summary)
        persistence.save_context(context_id, context_data)
        return history + [{"role": "user", "parts": [{"text": prompt}]}]

    def update_context(self, prompt: str, response_text: str, context_id: str):
        # ... (this method is unchanged) ...
        if not persistence.context_exists(context_id): return
        context_data = persistence.load_context(context_id)
        self.context_strategy.update_state(prompt, response_text, context_data)
        persistence.save_context(context_id, context_data)
        print(f"--- Context '{context_id}' updated. ---")