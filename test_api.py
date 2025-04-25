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
        result = remote_graph.invoke({
  "node": "root",
  "tasks": [],
  "chat_history": [],
  "real_chat_history": [
    {
      "type": "ai",
      "content": "Does the care recipient live in a private residence home or in a care facility?"
    },
    {
      "type": "human",
      "content": "He is living in an apartment, alone by himself"
    }
  ],
  "last_step": "start",
  "current_tree": "HousingAssessmentTree",
  "completed_whole_process": False,
  "care_recipient": {
    "address": "11650 National Boulevard, Los Angeles, California 90064, United States",
    "dateOfBirth": "1954-04-11",
    "dependentStatus": "Not a child/dependent",
    "firstName": "Yiming",
    "gender": "Male",
    "isSelf": "false",
    "lastName": "Gong",
    "legalName": "Yiming Gong",
    "pronouns": "he/him/his",
    "relationship": "dad",
    "veteranStatus": "Veteran"
  }
}, config=config)
        print("question: ", result["question"])
        print("real_chat_history: ", result["real_chat_history"])
        print("chat_history: ", result["chat_history"])
        print("completed_whole_process: ", result["completed_whole_process"])
    except GraphInterrupt as e:
        print(e)

if __name__ == "__main__":
    # asyncio.run(main())
    main()


