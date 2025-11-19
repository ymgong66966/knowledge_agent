from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
import uuid
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

def simplify_messages(messages):
    """Extract only content and type from messages"""
    return [{"type": msg.get("type"), "content": msg.get("content")} for msg in messages]

def main():
    # Configure your remote graph
    url = "https://ht-impressionable-sector-52-b5026c325a085ad2a0624561ba2fc6ff.us.langgraph.app"
    api_key = os.getenv("LANGCHAIN_API_KEY")
    graph_name = "onboarding_agent"

    # Initialize clients with API key
    client = get_client(url=url, api_key=api_key)
    sync_client = get_sync_client(url=url, api_key=api_key)
    thread = sync_client.threads.create()
    config = {"configurable": {"thread_id": thread["thread_id"]}}
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)
    # Start conversation
    try:
        result = remote_graph.invoke({
            "node": "root",
            "tasks": [],
            "chat_history": [],
            "veteranStatus": "Veteran",
            "real_chat_history": [
                {
                    "type": "ai",
                    "content": "Hello, @caregiverFirstName! Welcome to WithCare. I'm your AI Care Navigator trained by licensed clinicians to support you 24/7. How are you today?"
                },
                {
                    "type": "human",
                    "content": "I do not feel very good today. I feel a bit down"
                }
            ],
            "last_step": "start",
            "current_tree": "IntroAssessmentTree",
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
        print("thread_id: ", thread["thread_id"])
        print("question: ", result["question"])
        print("real_chat_history: ", simplify_messages(result["real_chat_history"]))
        print("chat_history: ", simplify_messages(result["chat_history"]))
        print("current_tree: ", result.get("current_tree"))
        print("last_step: ", result.get("last_step"))
    except GraphInterrupt as e:
        print(e)

if __name__ == "__main__":
    # asyncio.run(main())
    main()