import os
import json
import time
from gemini_manager.core import (
    GeminiManager,
    SimpleContextStrategy,
    RollingSummaryStrategy
)

# --- Test Suite Configuration ---
CONTEXT_LIFECYCLE_NAME = "test_lifecycle"
CONTEXT_SIMPLE_FLOW = "test_simple_flow"
CONTEXT_SUMMARY_FLOW = "test_summary_flow"
CONTEXT_BAD_INPUTS = "test_bad_inputs"


# --- Helper Functions ---
def run_test(test_function):
    """Decorator to run a test with setup and teardown."""
    print(f"\n{'='*20} RUNNING: {test_function.__name__} {'='*20}")
    try:
        test_function()
        print(f"--- RESULT: PASSED ---")
    except Exception as e:
        print(f"--- RESULT: FAILED ---")
        print(f"--- ERROR: {type(e).__name__}: {e} ---")
    finally:
        # Clean up specific contexts created by the test
        cleanup_specific_contexts([
            CONTEXT_LIFECYCLE_NAME,
            CONTEXT_SIMPLE_FLOW,
            CONTEXT_SUMMARY_FLOW,
            CONTEXT_BAD_INPUTS,
        ])

def cleanup_specific_contexts(context_list):
    """Deletes specific context files."""
    if not os.path.exists("contexts"):
        return
    for context_name in context_list:
        file_path = os.path.join("contexts", f"{context_name}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

def full_cleanup():
    """Deletes all .json files in the contexts directory."""
    print("\n--- Performing full cleanup of context files ---")
    if not os.path.exists("contexts"):
        os.makedirs("contexts")
        return
    for filename in os.listdir("contexts"):
        if filename.endswith(".json"):
            os.remove(os.path.join("contexts", filename))
    print("--- Cleanup complete ---")


# --- Test Cases ---

@run_test
def test_context_lifecycle_management():
    """Tests creation, duplicate creation, listing, and deletion."""
    manager = GeminiManager()
    
    # 1. Create a context
    print("Step 1: Creating a new context...")
    manager.create_context(CONTEXT_LIFECYCLE_NAME)
    assert CONTEXT_LIFECYCLE_NAME in manager.list_contexts(), "Context was not created."
    print("-> OK")

    # 2. Attempt to create a duplicate
    print("Step 2: Attempting to create a duplicate context...")
    try:
        manager.create_context(CONTEXT_LIFECYCLE_NAME)
        assert False, "Should have raised FileExistsError"
    except FileExistsError:
        print("-> OK: Correctly raised FileExistsError.")

    # 3. Delete the context
    print("Step 3: Deleting the context...")
    manager.delete_context(CONTEXT_LIFECYCLE_NAME)
    assert CONTEXT_LIFECYCLE_NAME not in manager.list_contexts(), "Context was not deleted."
    print("-> OK")

@run_test
def test_prepare_contents_for_nonexistent_context():
    """Tests that prepare_contents fails gracefully if context doesn't exist."""
    manager = GeminiManager()
    print("Attempting to prepare contents for a non-existent context...")
    try:
        manager.prepare_contents("A prompt", "i_do_not_exist")
        assert False, "Should have raised FileNotFoundError."
    except FileNotFoundError as e:
        assert "not found" in str(e), "Error message is not as expected."
        print(f"-> OK: Correctly raised FileNotFoundError.")

@run_test
def test_simple_conversation_flow():
    """Tests a standard multi-turn conversation with SimpleContextStrategy."""
    manager = GeminiManager(context_strategy=SimpleContextStrategy())
    manager.create_context(CONTEXT_SIMPLE_FLOW)

    # Turn 1
    print("Turn 1: Sending initial fact...")
    prompt1 = "The secret ingredient is cinnamon."
    contents1 = manager.prepare_contents(prompt1, CONTEXT_SIMPLE_FLOW)
    # In a real app, you'd make an API call here. We'll simulate it.
    response1_text = "I will remember that."
    manager.update_context(prompt1, response1_text, CONTEXT_SIMPLE_FLOW)

    # Verify context file
    with open(os.path.join("contexts", f"{CONTEXT_SIMPLE_FLOW}.json")) as f:
        data = json.load(f)
    assert len(data) == 2, "Context file should have 2 entries (user/model)."
    assert data[0]["parts"][0]["text"] == prompt1, "User prompt was not saved correctly."
    print("-> OK: Context for turn 1 saved correctly.")

    # Turn 2
    print("Turn 2: Asking a follow-up question...")
    prompt2 = "What is the secret ingredient?"
    contents2 = manager.prepare_contents(prompt2, CONTEXT_SIMPLE_FLOW)
    assert len(contents2) == 3, "Prepared contents should include history (2) + new prompt (1)."
    assert contents2[0]["parts"][0]["text"] == prompt1, "History was not loaded correctly."
    print("-> OK: Context from turn 1 was correctly prepared for turn 2.")
    
@run_test
def test_rolling_summary_flow():
    """Tests the full lifecycle of the RollingSummaryStrategy."""
    # Threshold=4 means summary happens AFTER the 2nd user prompt is processed.
    strategy = RollingSummaryStrategy(summary_threshold=4)
    manager = GeminiManager(context_strategy=strategy)
    manager.create_context(CONTEXT_SUMMARY_FLOW)

    # Turn 1 (History length: 2)
    print("Turn 1: Well below summary threshold...")
    manager.update_context("My name is Jane.", "Nice to meet you Jane.", CONTEXT_SUMMARY_FLOW)
    contents1 = manager.prepare_contents("A prompt", CONTEXT_SUMMARY_FLOW)
    assert "summary" not in str(contents1), "Summary should not be created yet."
    print("-> OK: No summary yet.")
    
    # Turn 2 (History length: 4) - this call to prepare_contents will trigger the summary.
    print("Turn 2: Reaching summary threshold...")
    manager.update_context("The password is 'fjord'.", "Got it.", CONTEXT_SUMMARY_FLOW)
    
    print("Preparing contents, which should trigger a summary API call...")
    # This call makes an API call internally for the summary
    prepared_contents = manager.prepare_contents("What's my name?", CONTEXT_SUMMARY_FLOW)
    
    # Verify the context file state
    with open(os.path.join("contexts", f"{CONTEXT_SUMMARY_FLOW}.json")) as f:
        data = json.load(f)
    
    assert data['summary'], "The summary field in the context file is empty."
    assert "Jane" in data['summary'] and "fjord" in data['summary'], "Summary content is missing key facts."
    assert len(data['history']) == 0, "History should be cleared after summarization."
    print("-> OK: Context file was correctly summarized and cleared.")
    
    # Verify the prepared contents for the current call
    assert "summary of our conversation" in prepared_contents[0]['parts'][0]['text'], "Prepared contents do not include the summary."
    print("-> OK: Next prompt is correctly prefixed with the new summary.")

@run_test
def test_input_variations():
    """Tests handling of empty and unicode inputs."""
    manager = GeminiManager()
    manager.create_context(CONTEXT_BAD_INPUTS)
    
    # 1. Empty prompt
    print("Step 1: Testing with an empty prompt...")
    prompt1 = ""
    contents1 = manager.prepare_contents(prompt1, CONTEXT_BAD_INPUTS)
    assert contents1[-1]['parts'][0]['text'] == "", "Empty prompt was not handled correctly."
    manager.update_context(prompt1, "You didn't say anything.", CONTEXT_BAD_INPUTS)
    print("-> OK")

    # 2. Unicode characters
    print("Step 2: Testing with unicode characters...")
    prompt2 = "My name is Schrödinger (Schrödinger's cat). My code is 你好."
    contents2 = manager.prepare_contents(prompt2, CONTEXT_BAD_INPUTS)
    assert "Schrödinger" in contents2[-1]['parts'][0]['text'], "Unicode string was mangled."
    assert "你好" in contents2[-1]['parts'][0]['text'], "Unicode string was mangled."
    manager.update_context(prompt2, "Interesting name and code!", CONTEXT_BAD_INPUTS)
    print("-> OK")
    
    # Verify file was saved correctly with unicode
    with open(os.path.join("contexts", f"{CONTEXT_BAD_INPUTS}.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert "Schrödinger" in data[2]['parts'][0]['text']
    print("-> OK: Unicode saved to file correctly.")

@run_test
def test_key_rotation_mechanic():
    """Confirms that get_client() cycles through the available keys."""
    # This test requires at least 3 keys in the .env file to be meaningful
    manager = GeminiManager()
    num_keys = len(manager.api_keys)
    if num_keys < 2:
        print("-> SKIPPED: This test is not meaningful with fewer than 2 API keys.")
        return
        
    print(f"Found {num_keys} API keys. Testing rotation...")
    
    # Get the initial order of keys
    initial_keys = list(manager.api_keys)
    
    # Cycle through the keys num_keys times
    used_keys = []
    for i in range(num_keys):
        client = manager.get_client() # This rotates the key
        # The key just used is now at the back of the deque
        used_keys.append(manager.api_keys[-1])

    # After one full cycle, the order should be the same as the start
    assert list(manager.api_keys) == initial_keys, "Key deque did not return to original state."
    # The set of used keys should match the initial set
    assert set(used_keys) == set(initial_keys), "Did not use all keys exactly once."
    print(f"-> OK: Cycled through all {num_keys} keys correctly.")


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        full_cleanup()
        test_context_lifecycle_management()
        test_prepare_contents_for_nonexistent_context()
        test_simple_conversation_flow()
        test_rolling_summary_flow()
        test_input_variations()
        test_key_rotation_mechanic()
    finally:
        full_cleanup()
        print("\nTest suite finished.")