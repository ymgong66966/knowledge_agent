from langgraph_sdk import get_sync_client, get_client
from langgraph.pregel.remote import RemoteGraph
from langgraph.errors import GraphInterrupt
from langchain_core.messages import AIMessage, HumanMessage
def main():
    # Configure your LOCAL dev server
    url = "http://localhost:2024"
    graph_name = "onboarding_agent"

    # Initialize clients (no API key needed for local dev)
    client = get_client(url=url)
    sync_client = get_sync_client(url=url)
    
    # Create a new thread
    thread = sync_client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Thread ID: {thread_id}")
    
    config = {"configurable": {"thread_id": thread_id}}
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)
    
    # Start conversation with your input
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
                    "content": "I do not feel good today. I feel a bit down"
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
            },
            "direct_record_answer": False
        }, config=config)
        
        print("\n" + "="*50)
        print("RESULT:", result)
        print("="*50)
        print(f"Thread ID: {thread_id}")
        print(f"\nQuestion: {result.get('question', 'N/A')}")
        print(f"\nCompleted whole process: {result.get('completed_whole_process', 'N/A')}")
        print(f"\nReal chat history length: {len(result.get('real_chat_history', []))}")
        print(f"\nChat history length: {len(result.get('chat_history', []))}")
        print(f"\nCurrent tree: {result.get('current_tree', 'N/A')}")
        print(f"\nLast step: {result.get('last_step', 'N/A')}")
        
        # Get full state
        state = remote_graph.get_state(config)
        print(f"\nFull state keys: {list(state.values.keys())}")
        
        return thread_id, result
        
    except GraphInterrupt as e:
        print(f"Graph interrupted: {e}")
        return thread_id, None
    except Exception as e:
        print(f"Error: {e}")
        return thread_id, None

if __name__ == "__main__":
    main()