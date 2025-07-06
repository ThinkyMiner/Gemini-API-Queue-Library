import os
import time
import json
from gemini_manager.core import (
    GeminiManager,
    SimpleContextStrategy,
    RollingSummaryStrategy,
    RetrievalAugmentedStrategy,
    persistence
)

# --- Test Suite Runner (Unchanged) ---
def run_test(test_function):
    """A decorator to run a test with standardized output and cleanup."""
    test_name = test_function.__name__
    print(f"\n{'='*20} RUNNING: {test_name} {'='*20}")
    try:
        test_function()
        print(f"--- RESULT: PASSED ---")
    except Exception as e:
        print(f"--- RESULT: FAILED ---")
        print(f"--- ERROR: {type(e).__name__}: {e} ---")
        raise
    finally:
        context_name = f"{test_name}_context"
        if os.path.exists(os.path.join("contexts", f"{context_name}.json")):
            os.remove(os.path.join("contexts", f"{context_name}.json"))
            print(f"--- Cleaned up '{context_name}.json' ---")


# --- Test Cases (These will be run automatically by the decorator) ---

@run_test
def test_simple_strategy_remembers_context():
    CONTEXT_NAME = "test_simple_strategy_remembers_context_context"
    manager = GeminiManager(context_strategy=SimpleContextStrategy())
    manager.create_context(CONTEXT_NAME)
    fact = "The launch code for Nebula is Tango-Charlie-Niner."
    manager.update_context(fact, "I will remember that.", CONTEXT_NAME)
    question = "What is the launch code for Nebula?"
    prepared_contents = manager.prepare_contents(question, CONTEXT_NAME)
    assert len(prepared_contents) == 3, "Prepared contents should have history (2) + new prompt (1)."
    assert fact in prepared_contents[0]["parts"][0]["text"], "The original fact is missing from the prepared history."
    print("-> OK: Simple strategy correctly recalled the conversation history.")


@run_test
def test_rolling_summary_strategy_summarizes_and_forgets():
    CONTEXT_NAME = "test_rolling_summary_strategy_summarizes_and_forgets_context"
    strategy = RollingSummaryStrategy(summary_threshold=2)
    manager = GeminiManager(context_strategy=strategy)
    manager.create_context(CONTEXT_NAME)
    fact1 = "The meeting is with 'CyberDyne Systems'."
    fact2 = "The primary topic is the 'Skynet' proposal."
    response1 = "Information noted."
    manager.update_context(f"{fact1} {fact2}", response1, CONTEXT_NAME)
    print("Preparing contents, which should trigger a summary...")
    question = "What is the main topic of our meeting?"
    prepared_contents = manager.prepare_contents(question, CONTEXT_NAME)
    context_data = persistence.load_context(CONTEXT_NAME)
    assert len(context_data['history']) == 0, "History list was not cleared after summarization."
    assert context_data['summary'], "Summary field is empty."
    assert "CyberDyne" in context_data['summary'] and "Skynet" in context_data['summary'], "Summary is missing key facts."
    print("-> OK: Context file was correctly updated with a summary.")
    assert "summary of our conversation" in prepared_contents[0]['parts'][0]['text'], "Prompt was not prefixed with the summary."
    print("-> OK: Summary was correctly injected into the next prompt.")


@run_test
def test_rag_strategy_retrieves_relevant_fact():
    CONTEXT_NAME = "test_rag_strategy_retrieves_relevant_fact_context"
    manager = GeminiManager(context_strategy=RetrievalAugmentedStrategy())
    manager.create_context(CONTEXT_NAME)
    unique_fact = "The secret ingredient for the 'Chronos' project is ytterbium-infused quartz."
    manager.update_context(unique_fact, "Fact stored.", CONTEXT_NAME)
    print("-> OK: Taught the model a unique fact.")
    manager.update_context("What's the weather today?", "It is sunny.", CONTEXT_NAME)
    manager.update_context("Tell me about the Roman Empire.", "It was a vast empire...", CONTEXT_NAME)
    print("-> OK: Added distraction conversation.")
    time.sleep(1)
    question = "What is the special ingredient for the Chronos project?"
    prepared_contents = manager.prepare_contents(question, CONTEXT_NAME)
    final_prompt = prepared_contents[0]['parts'][0]['text']
    assert "ytterbium-infused quartz" in final_prompt, "RAG failed to retrieve and inject the relevant fact."
    assert "Roman Empire" not in final_prompt, "RAG incorrectly retrieved irrelevant information."
    print("-> OK: RAG correctly augmented the prompt with the specific fact.")


# --- Main Execution Block (FIXED) ---
if __name__ == "__main__":
    print("\n\nStarting Context Strategy Test Suite...")
    
    # A single cleanup before all tests start. The decorator handles cleanup after each test.
    if os.path.exists("contexts"):
        for f in os.listdir("contexts"):
            if f.endswith(".json"):
                os.remove(os.path.join("contexts", f))
    
    # The tests above are automatically run when Python loads the file because of the @run_test decorator.
    # This block will only be executed AFTER all the tests have finished running.
    # Therefore, we do not call them again here.
    
    print("\nAll strategy tests have completed.")