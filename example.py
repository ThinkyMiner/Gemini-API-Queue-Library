from gemini_manager.core import GeminiManager, SimpleContextStrategy

def main():
    print("--- Transparent Gemini Helper Demo ---")
    
    manager = GeminiManager(context_strategy=SimpleContextStrategy())
    CONTEXT_NAME = "my-transparent-chat"

    # 1. Manually create the context
    if not manager.list_contexts() or CONTEXT_NAME not in manager.list_contexts():
        print(f"Creating context '{CONTEXT_NAME}'...")
        manager.create_context(CONTEXT_NAME)
    else:
        print(f"Using existing context '{CONTEXT_NAME}'.")

    # --- Have a conversation turn ---
    prompt_1 = "My favorite color is blue. My lucky number is 7."
    print(f"\nUser: {prompt_1}")

    # 2. Get a client with a rotated key
    client_1 = manager.get_client()

    # 3. Prepare the contents for the API call
    contents_1 = manager.prepare_contents(prompt_1, CONTEXT_NAME)
    
    # 4. Make your own, direct API call
    print("--- Making direct API call to Google... ---")
    response_1 = client_1.generate_content(
        model=f"models/{manager.context_strategy.model_name}",
        contents=contents_1,
        # You can add other parameters here freely!
        # generation_config={"temperature": 0.8}
    )
    response_text_1 = response_1.candidates[0].content.parts[0].text
    print(f"Gemini: {response_text_1}")

    # 5. Update the context with the result
    manager.update_context(prompt_1, response_text_1, CONTEXT_NAME)


    # --- Have a second turn to test context ---
    prompt_2 = "What is my lucky number?"
    print(f"\nUser: {prompt_2}")

    client_2 = manager.get_client()
    contents_2 = manager.prepare_contents(prompt_2, CONTEXT_NAME)
    
    print("--- Making direct API call to Google... ---")
    response_2 = client_2.generate_content(
        model=f"models/{manager.context_strategy.model_name}",
        contents=contents_2
    )
    response_text_2 = response_2.candidates[0].content.parts[0].text
    print(f"Gemini: {response_text_2}")

    manager.update_context(prompt_2, response_text_2, CONTEXT_NAME)

    # Cleanup
    manager.delete_context(CONTEXT_NAME)


if __name__ == "__main__":
    main()