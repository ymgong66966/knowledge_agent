from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
# Use the thread_id from local_test_api.py
thread_id = "628d02d5-ee61-402e-8681-87ba0a78531e"  # Replace with your actual thread_id
def simplify_messages(messages):
    """Extract only content and type from messages"""
    return [{"type": msg.get("type"), "content": msg.get("content")} for msg in messages]
config = {"configurable": {"thread_id": thread_id}}
print(f"Using Thread ID: {thread_id}")

def main():
    # Configure your LOCAL dev server
    url = "https://ht-impressionable-sector-52-b5026c325a085ad2a0624561ba2fc6ff.us.langgraph.app"
    api_key = os.getenv("LANGCHAIN_API_KEY")
    graph_name = "onboarding_agent"

    # Initialize clients (no API key needed for local dev)
    client = get_client(url=url, api_key=api_key)
    sync_client = get_sync_client(url=url, api_key=api_key)
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)

    try:
        # Get current state
        state = remote_graph.get_state(config)
        print("\n" + "="*50)
        print("CURRENT STATE:")
        print("="*50)
        print(f"Current question: {state.values.get('question', 'N/A')}")
        print(f"Real chat history length: {len(state.values.get('real_chat_history', []))}")
        print(f"Current tree: {state.values.get('current_tree', 'N/A')}")
        
        # Prepare the new message
        new_message = HumanMessage(content="i need to get my dad's hospital records")
        # I need to figure out the medicaid benefits for my dad
        
        # Update state with new message
        updated_state = dict(state.values)  # Make a copy of the current state
        updated_state["real_chat_history"] = state.values["real_chat_history"] + [new_message]
        
        print("\n" + "="*50)
        print("SENDING NEW MESSAGE:")
        print("="*50)
        print(f"Message: {new_message.content}")
        
        # Invoke the graph with the updated state using SAME thread_id
        result = remote_graph.invoke(updated_state, config=config)
        
        # Get updated state
        new_state = remote_graph.get_state(config)
        
        print("\n" + "="*50)
        print("NEW STATE AFTER RESPONSE:", new_state)
        print("="*50)
        print(f"Thread ID: {thread_id}")
        print(f"\nNew question: {new_state.values.get('question', 'N/A')}")
        print("real_chat_history: ", simplify_messages(result["real_chat_history"]))
        print("chat_history: ", simplify_messages(result["chat_history"]))
        print(f"\nCurrent tree: {new_state.values.get('current_tree', 'N/A')}")
        print(f"\nLast step: {new_state.values.get('last_step', 'N/A')}")
        
        return result
        
    except GraphInterrupt as e:
        print(f"Graph interrupted: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()