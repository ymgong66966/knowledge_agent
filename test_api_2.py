from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
import uuid
import asyncio
from langchain_core.messages import HumanMessage
# Create a new thread ID
thread_id = "6f8b0b5f-bdf6-47e6-9ebd-e4e537ec2d74"

config = {"configurable": {"thread_id": thread_id}}
print(f"Thread ID: {thread_id}")

def main():
    # Configure your remote graph
    url = "https://ht-impressionable-sector-52-b5026c325a085ad2a0624561ba2fc6ff.us.langgraph.app"
    api_key = "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"
    graph_name = "onboarding_agent"

    # Initialize clients with API key
    client = get_client(url=url, api_key=api_key)
    sync_client = get_sync_client(url=url, api_key=api_key)
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)

    # Start conversation
    try:
        state = remote_graph.get_state(config)
        # Prepare the new message
        new_message = HumanMessage(content="No")
        # Update both real_chat_history and chat_history
        updated_state = dict(state.values)  # Make a copy of the current state
        print("updated_state: ", updated_state)
        # the message list
        print("chat_history: ", updated_state["chat_history"])


        updated_state["real_chat_history"] = state.values["real_chat_history"] + [new_message]
        
        # Now invoke the graph with the updated state
        new_thread_id = str(uuid.uuid4())
        print("new thread id: ", new_thread_id)
        new_config = {"configurable": {"thread_id": new_thread_id}}
        result = remote_graph.invoke(updated_state, config = new_config)
        state = remote_graph.get_state(new_config)

        print("new chat history: ", state.values["chat_history"])
        print("question: ", state.values["question"])
    except GraphInterrupt as e:
        print(e)

if __name__ == "__main__":
    # asyncio.run(main())
    main()


