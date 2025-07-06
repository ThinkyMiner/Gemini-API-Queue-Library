from abc import ABC, abstractmethod
from google.ai import generativelanguage_v1beta as genai_services

class BaseContextStrategy(ABC):
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
    def get_initial_state(self): pass
    def prepare_history(self, context_data: any, client: any): pass
    def update_state(self, prompt: str, response: str, context_data: any): pass


class SimpleContextStrategy(BaseContextStrategy):
    def get_initial_state(self):
        return []
    def prepare_history(self, context_data: list, client=None):
        return context_data
    def update_state(self, prompt: str, response: str, context_data: list):
        context_data.append({"role": "user", "parts": [{"text": prompt}]})
        context_data.append({"role": "model", "parts": [{"text": response}]})


class RollingSummaryStrategy(BaseContextStrategy):
    def __init__(self, model_name="gemini-1.5-flash", summary_threshold=6):
        super().__init__(model_name)
        self.summary_threshold = summary_threshold

    def get_initial_state(self):
        return {'summary': '', 'history': []}

    def _summarize(self, text_to_summarize: str, client: genai_services.GenerativeServiceClient):
        print("\n--- (Helper) Making API call to summarize context... ---")
        response = client.generate_content(
            model=f"models/{self.model_name}",
            contents=[{"role": "user", "parts": [{"text": f"Concisely summarize this conversation:\n\n{text_to_summarize}"}]}]
        )
        return response.candidates[0].content.parts[0].text

    def prepare_history(self, context_data: dict, client: genai_services.GenerativeServiceClient):
        # Check if we need to summarize BEFORE preparing the history
        if len(context_data['history']) >= self.summary_threshold:
            history_text = "\n".join([f"{item['role']}: {item['parts'][0]['text']}" for item in context_data['history']])
            full_text = f"{context_data.get('summary', '')}\n{history_text}"
            
            new_summary = self._summarize(full_text, client)
            
            context_data['summary'] = new_summary
            context_data['history'] = [] # Clear history after summarization
        
        # Now, construct the history for the main call
        history = []
        if context_data['summary']:
            summary_context = f"This is a summary of our conversation so far: {context_data['summary']}"
            history.append({"role": "user", "parts": [{"text": summary_context}]})
            history.append({"role": "model", "parts": [{"text": "Understood. Let's continue."}]})
        history.extend(context_data['history'])
        return history

    def update_state(self, prompt: str, response: str, context_data: dict):
        context_data['history'].append({"role": "user", "parts": [{"text": prompt}]})
        context_data['history'].append({"role": "model", "parts": [{"text": response}]})