from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
import uuid
import asyncio
# Create a new thread ID
thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}
print(f"Thread ID: {thread_id}")

async def main():
    # Configure your remote graph
    url = "https://ht-impressionable-sector-52-b5026c325a085ad2a0624561ba2fc6ff.us.langgraph.app"
    api_key = "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"
    graph_name = "agent"

    # Initialize clients with API key
    client = get_client(url=url, api_key=api_key)
    sync_client = get_sync_client(url=url, api_key=api_key)
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)

    # Start conversation
    try:
        result = await remote_graph.ainvoke({
                "question": "what do you know about veteran benefits?",
                "domain": "financial",
            }, config=config)
        print(result['final_response'])
    except GraphInterrupt as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())

