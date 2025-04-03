from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
import uuid

# Configure your remote graph
url = "https://ht-impressionable-sector-52-b5026c325a085ad2a0624561ba2fc6ff.us.langgraph.app"
api_key = "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"
graph_name = "agent"

# Initialize clients with API key
client = get_client(url=url, api_key=api_key)
sync_client = get_sync_client(url=url, api_key=api_key)
remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)

# Create a new thread ID
thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}
print(f"Thread ID: {thread_id}")

try:
    # Start conversation
    print("\nStarting conversation...")
    result = remote_graph.invoke({
            "question": "I'm feeling sad",
            "domain": "financial",
            "conversation_history": ""
        }, config=config)
    print("Result:", result['final_response'])

    
except GraphInterrupt as e:
    print("Graph interrupted:", e.args[0])
# print("\nInterrupted - Address needed")
# # Resume with address using Command
# result = remote_graph.invoke(
#     Command(resume={"answer": "222 east pearson street, 60611"}),
#     config
# )
# print("\nResult after providing address:", result)

# Get final state
# snapshot = remote_graph.get_state(config)
# print("\nFinal state:", snapshot)
