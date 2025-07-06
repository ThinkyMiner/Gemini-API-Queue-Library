from abc import ABC, abstractmethod
from google.ai import generativelanguage_v1beta as genai_services
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid

class BaseContextStrategy(ABC):
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
    def get_initial_state(self): pass
    def prepare_history(self, context_data: any, **kwargs): pass
    def update_state(self, prompt: str, response: str, context_data: any, **kwargs): pass


class SimpleContextStrategy(BaseContextStrategy):
    def get_initial_state(self): return []
    def prepare_history(self, context_data: list, **kwargs): return context_data
    def update_state(self, prompt: str, response: str, context_data: list, **kwargs):
        context_data.append({"role": "user", "parts": [{"text": prompt}]})
        context_data.append({"role": "model", "parts": [{"text": response}]})

class RollingSummaryStrategy(BaseContextStrategy):
    # ... (this class is unchanged from before) ...
    def __init__(self, model_name="gemini-1.5-flash", summary_threshold=6):
        super().__init__(model_name)
        self.summary_threshold = summary_threshold
    def get_initial_state(self): return {'summary': '', 'history': []}
    def _summarize(self, text_to_summarize, client):
        print("\n--- (Helper) Making API call to summarize context... ---")
        response = client.generate_content(model=f"models/{self.model_name}", contents=[{"role": "user", "parts": [{"text": f"Concisely summarize this conversation:\n\n{text_to_summarize}"}]}])
        return response.candidates[0].content.parts[0].text
    def prepare_history(self, context_data: dict, **kwargs):
        client = kwargs.get("client")
        if len(context_data['history']) >= self.summary_threshold:
            history_text = "\n".join([f"{item['role']}: {item['parts'][0]['text']}" for item in context_data['history']])
            full_text = f"{context_data.get('summary', '')}\n{history_text}"
            new_summary = self._summarize(full_text, client)
            context_data['summary'] = new_summary
            context_data['history'] = []
        history = []
        if context_data['summary']:
            history.append({"role": "user", "parts": [{"text": f"This is a summary of our conversation so far: {context_data['summary']}"}]})
            history.append({"role": "model", "parts": [{"text": "Understood. Let's continue."}]})
        history.extend(context_data['history'])
        return history
    def update_state(self, prompt: str, response: str, context_data: dict, **kwargs):
        context_data['history'].append({"role": "user", "parts": [{"text": prompt}]})
        context_data['history'].append({"role": "model", "parts": [{"text": response}]})

class RetrievalAugmentedStrategy(BaseContextStrategy):
    """Uses a vector database for context."""
    def __init__(self, model_name="gemini-1.5-flash", top_k=3):
        super().__init__(model_name)
        print("Initializing RAG: Loading sentence-transformer model...")
        # Use a file-based Qdrant for persistence across test runs if needed, but memory is fine
        self.qdrant_client = QdrantClient(":memory:")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.top_k = top_k
        print("RAG Initialized.")
    def get_initial_state(self):
        # The JSON file only serves to prove the context exists.
        # We also store the unique collection name for the vector DB.
        return {"strategy": "rag", "collection_name": str(uuid.uuid4())}

    def _get_or_create_collection(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception:
            self.qdrant_client.recreate_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=self.embedding_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE))

    def prepare_history(self, context_data: dict, **kwargs):
        # RAG doesn't use linear history. It prepares a special prompt.
        # The actual prompt augmentation happens in the main manager.
        return [] # Return an empty history, as it's not used.

    def update_state(self, prompt: str, response: str, context_data: dict, **kwargs):
        collection_name = context_data['collection_name']
        self._get_or_create_collection(collection_name)
        interaction_text = f"User asked: {prompt}\nAI responded: {response}"
        vector = self.embedding_model.encode(interaction_text).tolist()
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": interaction_text})],
            wait=True
        )