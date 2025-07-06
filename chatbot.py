import os
import sys
from gemini_manager.core import (
    GeminiManager,
    SimpleContextStrategy,
    RollingSummaryStrategy,
    RetrievalAugmentedStrategy
)

def select_strategy():
    """Lets the user choose which context management strategy to use."""
    print("Welcome to the Gemini Chatbot!")
    print("Please select a context management strategy:")
    print("1. Simple Strategy (Standard conversation, can hit token limits)")
    print("2. Rolling Summary Strategy (Summarizes long conversations, uses extra API calls)")
    print("3. Retrieval Augmented (RAG) Strategy (Best for recalling specific facts)")

    while True:
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            return GeminiManager(context_strategy=SimpleContextStrategy())
        elif choice == '2':
            # We can configure the threshold here if needed
            strategy = RollingSummaryStrategy(summary_threshold=6)
            return GeminiManager(context_strategy=strategy)
        elif choice == '3':
            return GeminiManager(context_strategy=RetrievalAugmentedStrategy())
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def select_context(manager: GeminiManager):
    """Lets the user create a new context or load an existing one."""
    print("\n--- Context Selection ---")
    
    contexts = manager.list_contexts()
    if contexts:
        print("Existing conversations (contexts):")
        for name in contexts:
            print(f"- {name}")
    else:
        print("No existing conversations found.")
    
    print("\nOptions:")
    print("[c] - Create a new conversation.")
    print("[l] - List existing conversations again.")
    print("Or type the name of an existing conversation to load it.")

    while True:
        choice = input("Your choice: ").strip()
        if choice.lower() == 'c':
            new_context_name = input("Enter a name for your new conversation: ").strip()
            if not new_context_name:
                print("Name cannot be empty.")
                continue
            try:
                manager.create_context(new_context_name)
                print(f"Conversation '{new_context_name}' created successfully.")
                return new_context_name
            except FileExistsError:
                print(f"Error: A conversation with the name '{new_context_name}' already exists.")

        elif choice.lower() == 'l':
            contexts = manager.list_contexts()
            if contexts:
                for name in contexts:
                    print(f"- {name}")
            else:
                print("No existing conversations found.")
        elif choice in contexts:
            print(f"Loading conversation '{choice}'...")
            return choice
        else:
            print(f"Invalid choice. Type 'c', 'l', or an existing conversation name.")


def print_help():
    """Prints the available commands."""
    print("\n--- Chatbot Commands ---")
    print("/new    - Start a new conversation or switch to a different one.")
    print("/exit   - Exit the chatbot.")
    print("/help   - Show this help message.")
    print("------------------------\n")

def chat_with_bot(manager: GeminiManager, context_id: str):
    """The main interactive chat loop."""
    print("\n--- Starting Chat ---")
    print_help()
    
    while True:
        prompt = input(f"You ({context_id}): ").strip()

        if not prompt:
            continue
        
        # Handle commands
        if prompt.lower() == '/exit':
            break
        if prompt.lower() == '/new':
            new_context_id = select_context(manager)
            if new_context_id:
                context_id = new_context_id
                print(f"\nSwitched to conversation '{context_id}'.")
            continue
        if prompt.lower() == '/help':
            print_help()
            continue
            
        try:
            # 1. Get the configured API client (with a rotated key)
            client = manager.get_client()

            # 2. Prepare the full conversation history + new prompt
            contents = manager.prepare_contents(prompt, context_id)
            
            # 3. Make the direct call to Google's API
            print("Gemini: Thinking...")
            model_name_to_use = f"models/{manager.context_strategy.model_name}"
            response = client.generate_content(
                model=model_name_to_use,
                contents=contents
            )
            
            # 4. Extract and print the response text
            if not response.candidates:
                # This can happen if the content is blocked by safety filters
                response_text = "I'm sorry, I can't provide a response to that."
            else:
                response_text = response.candidates[0].content.parts[0].text
            
            print(f"Gemini: {response_text}")

            # 5. Update our local context file with the latest turn
            manager.update_context(prompt, response_text, context_id)

        except FileNotFoundError as e:
            print(f"ERROR: {e}. This shouldn't happen in the chat loop. Exiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please try again.")


def main():
    """Main function to run the chatbot application."""
    manager = select_strategy()
    context_id = select_context(manager)
    if context_id:
        chat_with_bot(manager, context_id)


if __name__ == "__main__":
    try:
        main()
        print("\nChatbot session ended. Goodbye!")
    except KeyboardInterrupt:
        print("\n\nCaught Ctrl+C. Exiting gracefully. Goodbye!")
    except Exception as e:
        print(f"\nA critical error occurred: {e}")