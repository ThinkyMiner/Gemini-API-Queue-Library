Of course. A great library deserves great documentation.

Here is a comprehensive `README.md` file that explains the architecture, the API, the different context strategies, and provides clear usage examples. You can save this directly in the root of your project.

---

# Gemini Advanced Context Manager

A Python library to supercharge Google's Gemini API with advanced context management and API key rotation.

This library acts as a transparent "helper" for the official Google Gemini SDK, allowing you to build stateful, multi-turn conversations while staying within free-tier rate limits. It is designed to give developers maximum flexibility by not hiding the underlying Google API calls.

## Core Problem This Solves

1.  **Stateless API**: Google's Gemini API is stateless. Each call is independent, making it difficult to build conversations that remember previous turns.
2.  **Rate Limiting**: Free-tier API keys have strict rate limits (e.g., requests per minute). A simple application can quickly hit this limit.

## Key Features

-   **Automatic API Key Rotation**: Distributes your API calls across multiple keys to avoid rate-limit errors.
-   **Persistent Conversations**: Saves conversation history to local files, allowing you to continue chats later.
-   **Multiple Context Strategies**: Choose the best context management method for your specific use case.
    -   **Simple**: Standard turn-by-turn history.
    -   **Rolling Summary**: Automatically summarizes older parts of the conversation to keep the context window small.
    -   **Retrieval-Augmented Generation (RAG)**: Uses a vector database to recall specific facts from very long conversations.
-   **Transparent Architecture**: You still make the final API call yourself, giving you full control over all of Google's parameters (`safety_settings`, `generation_config`, etc.).

---

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <(https://github.com/ThinkyMiner/gemini_queue_lib.git)>
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Keys**
    Create a file named `.env` in the root directory of the project. Open it and add your Gemini API keys, separated by commas.

    **`.env` file:**
    ```
    GEMINI_API_KEYS=your_first_api_key,your_second_api_key,your_third_api_key
    ```

---

## The Core Architecture (How It Works)

This library does **not** make the final API call for you. Instead, it prepares everything you need, giving you full control. The workflow for every conversational turn is a simple four-step process:

1.  **`manager.get_client()`**: Get a pre-configured Google API client. The manager handles rotating the API key for you in this step.
2.  **`manager.prepare_contents()`**: Give it your new prompt and a conversation ID. It loads the relevant history from files, manages the context according to your chosen strategy (e.g., creates a summary, does a vector search), and returns a perfectly formatted `contents` list.
3.  **`client.generate_content()`**: You make the **direct** call to Google's API using the `client` and `contents` from the previous steps.
4.  **`manager.update_context()`**: After you get a response, you pass the original prompt and the response text back to the manager, which saves the new turn to the conversation history file.

![Architecture Diagram](./files/architecture.png')

![Thought Process](./files/thought_process.png')

---

## Quick Start: A Simple Conversation

This example shows the basic workflow using the default `SimpleContextStrategy`.

```python
from gemini_manager import GeminiManager, SimpleContextStrategy

# Initialize the manager with your chosen strategy
manager = GeminiManager(context_strategy=SimpleContextStrategy())
CONTEXT_NAME = "my_first_chat"

# --- Turn 1 ---

# 1. Create a named conversation if it doesn't exist
try:
    manager.create_context(CONTEXT_NAME)
    print(f"Conversation '{CONTEXT_NAME}' created.")
except FileExistsError:
    print(f"Conversation '{CONTEXT_NAME}' already exists. Continuing...")

prompt1 = "Hi there! An important fact to remember is that Project Phoenix uses the color red."

# 2. Get a client with a rotated API key
client1 = manager.get_client()

# 3. Prepare the conversation history + new prompt
contents1 = manager.prepare_contents(prompt1, CONTEXT_NAME)

# 4. Make the direct API call
print("You: " + prompt1)
print("Gemini: Thinking...")
response1 = client1.generate_content(
    model=f"models/{manager.context_strategy.model_name}",
    contents=contents1
)
response_text1 = response1.candidates[0].content.parts[0].text
print("Gemini: " + response_text1)

# 5. Save the result
manager.update_context(prompt1, response_text1, CONTEXT_NAME)


# --- Turn 2 ---

prompt2 = "What color is associated with Project Phoenix?"

client2 = manager.get_client()
contents2 = manager.prepare_contents(prompt2, CONTEXT_NAME)

print("\nYou: " + prompt2)
print("Gemini: Thinking...")
response2 = client2.generate_content(
    model=f"models/{manager.context_strategy.model_name}",
    contents=contents2
)
response_text2 = response2.candidates[0].content.parts[0].text
print("Gemini: " + response_text2)

manager.update_context(prompt2, response_text2, CONTEXT_NAME)
```

---

## API Reference (`GeminiManager`)

### `__init__(context_strategy)`
Initializes the manager.
-   **`context_strategy`**: An instance of a strategy class (e.g., `SimpleContextStrategy()`). Defaults to `SimpleContextStrategy`.

### `get_client()`
Rotates the API key queue and returns a configured `GenerativeServiceClient` instance for you to use.
-   **Returns**: A Google API client object.

### `prepare_contents(prompt, context_id)`
The core function for preparing the conversation.
-   **`prompt`** (str): The new user message.
-   **`context_id`** (str): The unique name of the conversation you want to load.
-   **Returns**: A `contents` list, ready to be passed to `client.generate_content()`.

### `update_context(prompt, response_text, context_id)`
Saves the latest turn to the conversation's history file.
-   **`prompt`** (str): The user prompt from the turn.
-   **`response_text`** (str): The model's response from the turn.
-   **`context_id`** (str): The name of the conversation to update.

### `create_context(context_id)`
Creates a new, empty conversation file.
-   **`context_id`** (str): The name for the new conversation. Raises `FileExistsError` if the name is already taken.

### `list_contexts()`
-   **Returns**: A list of strings containing the names of all existing conversations.

### `delete_context(context_id)`
-   Deletes the conversation file associated with the given `context_id`.

---

## Context Management Strategies (The Modes)

You choose a strategy when you first initialize the `GeminiManager`. Each one has unique strengths and trade-offs.

### 1. `SimpleContextStrategy`
This is the default and most basic strategy.

-   **How it works**: It keeps a running list of every single turn (user and model) in the conversation history. On every new prompt, it sends the *entire* history to the API.
-   **When to use it**: Perfect for short-to-medium length conversations, simple Q&A bots, or any application where perfect, granular recall of the most recent turns is essential.
-   **Limitations**: It will eventually hit the Gemini API's token limit on very long conversations, resulting in an error. It is not suitable for conversations that need to last for hundreds of turns.
-   **Usage**:
    ```python
    from gemini_manager import GeminiManager, SimpleContextStrategy
    manager = GeminiManager(context_strategy=SimpleContextStrategy())
    ```

### 2. `RollingSummaryStrategy`
This strategy is designed for long-running conversations.

-   **How it works**: When the conversation history reaches a certain length (the `summary_threshold`), this strategy automatically makes a "helper" API call to summarize the existing history. It then *replaces* that history with the summary. Future turns are prefixed with "This is a summary of our conversation so far: ..."
-   **When to use it**: Ideal for chatbots acting as assistants, support bots, or long-term planners, where the general gist of the past is more important than the exact wording of a prompt from 50 turns ago.
-   **Limitations**: It uses extra API calls for summarization, which can increase costs. The summarization process can also lose fine-grained details or specific facts.
-   **Usage**:
    ```python
    from gemini_manager import GeminiManager, RollingSummaryStrategy
    # Summarize after every 10 turns (5 user, 5 model)
    strategy = RollingSummaryStrategy(summary_threshold=10)
    manager = GeminiManager(context_strategy=strategy)
    ```

### 3. `RetrievalAugmentedStrategy (RAG)`
This is the most advanced strategy, designed for perfect recall of specific information over extremely long periods.

-   **How it works**: It does not keep a linear history. Instead, it saves every conversational turn as an independent "memory" in a vector database. When you ask a new question, it first searches the database for the most relevant memories (e.g., facts, past questions) and injects only those into the prompt.
-   **When to use it**: Perfect for building a personal knowledge base, a documentation assistant, or any application where you need to recall specific, buried facts like "What was the project code I mentioned three weeks ago?".
-   **Limitations**: It is more computationally intensive and depends on external libraries (`sentence-transformers`, `qdrant-client`). Its effectiveness depends on the quality of the semantic search.
-   **Usage**:
    ```python
    from gemini_manager import GeminiManager, RetrievalAugmentedStrategy
    # Retrieve the top 3 most relevant memories for each prompt
    strategy = RetrievalAugmentedStrategy(top_k=3)
    manager = GeminiManager(context_strategy=strategy)
    ```

---

## Running the Tools

### Interactive Chatbot
To manually test all strategies and features, run the provided chatbot application.
```bash
python chatbot.py
```

### Test Suite
To verify that the library is working correctly on your system, you can run the test suites.
```bash
# Run the core functionality tests
python comprehensive_tests.py

# Run the strategy-specific tests
python test_strategies.py
```