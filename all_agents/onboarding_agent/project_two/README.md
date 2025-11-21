# Onboarding Agent – Project Two

This README documents two important issues we hit recently and how they were fixed, so future changes don’t re‑introduce them.

---

## 1. LangChain / LangGraph tracing (V1 vs V2)

### Problem

- Newer versions of `langchain-core` and LangGraph use **LangSmith tracing v2**.
- Our environment was still configured for **LangChainTracerV1**, which is no longer supported.
- This caused runtime errors when running the graph, for example:

  ```text
  RuntimeError: Tracing using LangChainTracerV1 is no longer supported. Please set the LANGCHAIN_TRACING_V2 environment variable to enable tracing instead.
  ```

### Resolution / Correct configuration

To avoid this error you must either **enable v2 tracing** or **disable tracing entirely**.

#### Option A – Enable v2 tracing (recommended when using LangSmith)

Set the following environment variable (either in `.env`, shell, or LangGraph Studio env config):

```bash
LANGCHAIN_TRACING_V2=true
```

If you use LangSmith, you should also have:

```bash
LANGCHAIN_API_KEY=...              # LangSmith API key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# Optional but recommended
LANGCHAIN_PROJECT=withcare-onboarding
```

Then restart `langgraph dev` / the runtime.

#### Option B – Turn tracing off completely

Unset any legacy v1 tracing env vars:

```bash
unset LANGCHAIN_TRACING
unset LC_TRACING
unset LANGCHAIN_HANDLER
unset LANGCHAIN_TRACING_V2  # or set it explicitly to "false"
```

Then restart `langgraph dev`. This disables tracing but removes the runtime error.

---

## 2. Pydantic v2 + GraphState + message types

### Original symptoms

When invoking the `onboarding_agent` graph we saw errors like:

```text
TypeError: BaseModel.validate() takes 2 positional arguments but 3 were given
```

This occurred while LangGraph was coercing input into the `GraphState` schema.

### Root cause

- The project is using **Pydantic v2**.
- `GraphState` was defined with message fields using **LangChain message models** as type hints:

  ```python
  from langchain_core.messages import BaseMessage, AnyMessage

  class GraphState(BaseModel):
      chat_history: list[BaseMessage]
      real_chat_history: list[AnyMessage]
  ```

- When we passed `AIMessage` / `HumanMessage` instances or other message objects into these fields, Pydantic v2 tried to **deep‑validate** them.
- That triggered legacy v1‑style validators somewhere in the LangChain message stack, leading to the `BaseModel.validate()` signature mismatch inside Pydantic.

### Fix: treat chat history as opaque lists

We stopped asking Pydantic to validate LangChain message models directly, and instead:

1. **Changed the types of history fields in `GraphState` to plain lists** (no Pydantic model types):

   ```python
   class GraphState(BaseModel):
       # ... other fields ...

       chat_history: list = Field(default=[])
       real_chat_history: list = Field(default=[])
       # assessment_answer can also be made a plain list if needed
       # assessment_answer: list = Field(default=[])
   ```

   The key is: these are now just `list` / `list[dict]` / `list[Any]`, so Pydantic v2 does **not** attempt deep validation of `AIMessage` / `HumanMessage` types.

2. **Kept the runtime representation as simple dicts for `real_chat_history`** when passing from the client:

   ```python
   "real_chat_history": [
       {"type": "ai", "content": "Hello, ..."},
       {"type": "human", "content": "I do not feel good today. I feel a bit down"},
   ]
   ```

   This is compatible with Pydantic v2 because `GraphState` just treats these as lists of plain Python dicts.

3. **Adjusted parsing logic in `parse_response` and related nodes** to work with dicts instead of LangChain message objects. For example:

   ```python
   # OLD: assuming LangChain HumanMessage / AIMessage objects
   for message in reversed(state.real_chat_history):
       if isinstance(message, HumanMessage):
           user_response = message.content + user_response
       elif isinstance(message, AIMessage):
           break
   ```

   became:

   ```python
   for message in reversed(state.real_chat_history):
       if message["type"] == "human":
           user_response = message["content"] + user_response
       elif message["type"] == "ai":
           break
   ```

### Key takeaways for future changes

- **Do not type `GraphState` fields as `list[BaseMessage]` or `list[AnyMessage]` when using Pydantic v2.** Use plain `list` / `list[dict]` / `list[Any]` instead, and handle message parsing manually.
- It is safe (and often simpler) to represent history in the state as a list of dicts with at least `{"type": ..., "content": ...}` and treat LangChain `AIMessage` / `HumanMessage` as a runtime concern at the edges.
- If you change the structure of `real_chat_history` or `chat_history`, make sure to update any nodes that iterate over them (e.g. `parse_response`, `assess_mental`, `ask_next_question`).

---

## 3. Client invocation patterns

Two useful invocation patterns we’ve verified to work with this setup:

1. **Starting a new thread with structured `real_chat_history` dicts** (from `local_test_api.py`):

   ```python
   result = remote_graph.invoke({
       "node": "root",
       "tasks": [],
       "chat_history": [],
       "veteranStatus": "Veteran",
       "real_chat_history": [
           {"type": "ai", "content": "Hello, ... How are you today?"},
           {"type": "human", "content": "I do not feel good today. I feel a bit down"},
       ],
       "last_step": "start",
       "current_tree": "IntroAssessmentTree",
       "care_recipient": {...},
       "direct_record_answer": False,
   }, config=config)
   ```

2. **Continuing a thread using `local_test_api_2.py`**:

   - Fetch the current state using `remote_graph.get_state(config)`.
   - Append a new `HumanMessage` at the edge, but keep `GraphState.real_chat_history` as list‑of‑dicts inside the graph.

This combination, together with the updated `GraphState` and parsing logic, avoids both the Pydantic validation error and the tracing runtime error.
