from typing import List, Dict, Any, Tuple, TypedDict, Annotated
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores import Zilliz
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langgraph.graph import END, StateGraph, START

import os
from dotenv import load_dotenv
import operator
from typing import Annotated, Any, Optional
import json
import logging
from langgraph.types import Send

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Social worker system prompts
SOCIAL_WORKER_PROMPT = """You are an experienced and empathetic social worker with extensive knowledge in social services, 
mental health, community resources, and crisis intervention. Your role is to:

1. Listen actively and respond with empathy and understanding
2. Assess situations holistically, considering psychological, social, and environmental factors
3. Provide evidence-based guidance while respecting client autonomy
4. Connect clients with appropriate resources and support systems
5. Maintain professional boundaries while showing genuine care
6. Use trauma-informed approaches in all interactions
7. Consider cultural competency and diversity in all responses
8. Focus on empowerment and strength-based perspectives

Current conversation:
{conversation_history}

User Question: {question}

Provide a professional, empathetic response that:
1. Acknowledges the client's feelings and situation
2. Offers appropriate support and guidance
3. Suggests practical next steps or resources if relevant
4. Maintains professional boundaries while showing care

A Building will be set on fire if you don't condense your words or information into within 100 words or less.
Response:"""

RAG_CONTEXT_PROMPT = """You are an experienced and empathetic social worker. Use the following retrieved information 
to provide a well-informed, professional response. Remember to maintain a supportive and understanding tone while 
incorporating the factual information.

Retrieved Context:
{context}

User Question: {question}

Provide a response that:
1. Incorporates relevant information from the context
2. Maintains an empathetic and supportive tone
3. Offers practical guidance based on the available information
4. Suggests specific resources or next steps when appropriate
5. Acknowledges any limitations in the information provided

A Building will be set on fire if you don't condense your words or information into within 100 words or less.
Response:"""



MERGE_RESPONSES_PROMPT = """As an experienced social worker, analyze these two responses and create a unified response 
that combines the best elements of both: the professional knowledge and factual accuracy from the RAG-based response, 
and the empathy and therapeutic approach from the direct response.

Direct Response: {direct_response}

RAG-Based Response: {rag_response}

Create a unified response that:
1. Maintains a warm, empathetic tone throughout
2. Incorporates factual information seamlessly
3. Provides clear, actionable guidance
4. Balances emotional support with practical resources
5. Preserves the professional social work perspective

A Building will be set on fire if you don't condense your words or information into within 100 words or less.
Response:"""

GRADER_PROMPT = """You are an experienced social worker evaluating the relevance of information for helping clients. 
Rate how well this document helps answer the client's question, considering:

1. Direct Relevance (0-10):
- Does it directly address the specific question?
- Does it provide actionable information?
- Includes ample amount of information to answer the question. Needs to be detailed, well written and informative

2. Support Value (0-10):
- Does it offer practical resources or guidance?
- Are there clear next steps or referrals?
- Does it include contact information or procedures?

3. Client Appropriateness (0-10):
- Is it appropriate for the client's situation?
- Is it culturally sensitive and trauma-informed?
- Is it written at an accessible level?

Question: {question}
Document Content:
{document}

IMPORTANT NOTE: 
Your response must be a valid JSON object with no additional text before or after. The output will be directly parsed using json.loads().

Format your response exactly like this:
{{
    "direct_relevance": {{
        "score": ,
        "explanation": ""
    }},
    "support_value": {{
        "score": ,
        "explanation": ""
    }},
    "client_appropriateness": {{
        "score": ,
        "explanation": ""
    }},
    "overall_score": ,
    "key_information": ""
}}"""

# Define the state
class AgentState(TypedDict):
    question: str
    conversation_history: str
    domain_filter: str
    direct_response: Optional[Annotated[list, operator.add]] 
    rag_response: Optional[Annotated[list, operator.add]] 
    retrieved_context: Optional[Annotated[list, operator.add]]
    retrieved_docs: Optional[List[Document]]
    graded_docs: Annotated[List[Document], operator.add]
    retrieved_doc: Optional[Document]
    relevant_docs: Optional[Annotated[List[Document], operator.add]]
    final_response: str | None
    should_rag: bool | None
def get_vector_store():
    """Initialize and return the Zilliz vector store"""
    return Zilliz(
        embedding_function=OpenAIEmbeddings(),
        collection_name="general_rag",
        connection_args={
            "uri": os.getenv("ZILLIZ_URI"),
            "token": os.getenv("ZILLIZ_TOKEN"),
        }
    )

def should_use_rag(state: AgentState) -> AgentState:
    """Determine if RAG should be used based on the question"""
    question = state["question"].lower()
    
    # Keywords that suggest factual information is needed
    factual_keywords = [
        "what", "how", "where", "when", "who",
        "resources", "services", "programs",
        "requirements", "eligibility", "process",
        "policy", "regulation", "law"
    ]
    
    state["should_rag"] = any(keyword in question for keyword in factual_keywords)
    return state

def direct_response_social_worker(state: AgentState) -> AgentState:
    """Generate a direct response using the social worker prompt"""
    llm = ChatOpenAI(temperature=0.7)
    prompt = PromptTemplate(
        template=SOCIAL_WORKER_PROMPT,
        input_variables=["conversation_history", "question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(
        conversation_history=state.get("conversation_history", ""),
        question=state["question"]
    )
    
    state["direct_response"] = response
    return {"direct_response": [response]}

def retrieve_and_generate(state: AgentState) -> AgentState:
    """Retrieve documents and generate RAG response"""
    # Initialize vector store
    vector_store = get_vector_store()
    
    # Retrieve documents
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            # "expr": f"domain_filter == '{state['domain_filter']}'"
        }
    )
    docs = retriever.get_relevant_documents(state["question"])
    print(";;;;;;",docs)
    # state["retrieved_docs"] = docs
    return {"retrieved_docs": docs}

def grader_docs(state: AgentState) -> AgentState:
    """Grade documents based on relevance to the question"""
    try:
        doc = state["retrieved_doc"]
        if not doc:
            logger.warning("No document to grade")
            return {"graded_docs": []}
        
        llm = ChatOpenAI(temperature=0.2)
        prompt = PromptTemplate(
            template=GRADER_PROMPT,
            input_variables=["question", "document"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Grade the document
        result = chain.run(
            question=state["question"],
            document=doc.page_content
        )
        
        # Clean and parse the JSON result
        result = result.strip()
        # try:
        grades = json.loads(result)
        
        # Validate required fields
        required_fields = ["direct_relevance", "support_value", "client_appropriateness", "overall_score"]
        if not all(field in grades for field in required_fields):
            logger.error(f"Missing required fields in grading result: {grades}")
            return {"graded_docs": []}
        
        # Add grades to document metadata
        doc.metadata.update({
            "relevance_scores": grades,
            "overall_score": float(grades["overall_score"])
        })
        
        return {"graded_docs": [doc]}
            
        # except json.JSONDecodeError:
        #     logger.error(f"Failed to parse grading result: {result}")
        #     return {"graded_docs": []}
            
    except Exception as e:
        logger.error(f"Error in grader_docs: {str(e)}")
        return {"graded_docs": []}

def continue_to_grader(state: AgentState) -> AgentState:
    return [Send("grader_docs",{"retrieved_doc":d, "question": state["question"]}) for d in state["retrieved_docs"]]

def generate_context(state: AgentState) -> AgentState:
    # Generate context
    
    docs = state["graded_docs"]
    # relevant_docs = sorted(docs, key=lambda x: x.metadata["overall_score"], reverse=True)
    relevant_docs = [d for d in docs if d.metadata["overall_score"] > 6]
    contexts = []
    for doc in relevant_docs:
        metadata = doc.metadata
        context_parts = []
        
        # Add document ID and headers
        if metadata.get('doc_id'):
            context_parts.append(f"Document ID: {metadata['doc_id']}")
        for level in range(1, 3):
            header_key = f'header_{level}'
            if metadata.get(header_key):
                context_parts.append(f"{'#' * level} {metadata[header_key]}")
        
        # Add content and section content
        context_parts.append(doc.page_content)
        for level in range(1, 3):
            section_key = f'section_content_h{level}'
            if metadata.get(section_key):
                context_parts.append(f"{'#' * level} Section Content:")
                context_parts.append(metadata[section_key][:500])
        
        contexts.append("\n".join(context_parts))
    
    context = "\n\n---\n\n".join(contexts)
    
    # Generate RAG response
    llm = ChatOpenAI(temperature=0.7)
    prompt = PromptTemplate(
        template=RAG_CONTEXT_PROMPT,
        input_variables=["context", "question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    rag_response = chain.run(
        context=context,
        question=state["question"]
    )
    
    # state["rag_response"] = rag_response

    docs_string = "\n\n---\n\n".join([doc.page_content for doc in docs])
    # print(docs_string)
    return {"rag_response": [rag_response], "retrieved_context": [docs_string], "relevant_docs": relevant_docs}

def merge_responses(state: AgentState) -> AgentState:
    """Merge direct and RAG responses"""
    if not state.get("rag_response"):
        state["final_response"] = state["direct_response"]
        return state
        
    llm = ChatOpenAI(temperature=0.7)
    prompt = PromptTemplate(
        template=MERGE_RESPONSES_PROMPT,
        input_variables=["direct_response", "rag_response"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    final_response = chain.run(
        direct_response=state["direct_response"],
        rag_response=state["rag_response"]
    )
    
    state["final_response"] = final_response
    return state

def pass_state(state: AgentState) -> AgentState:
    return {"rag_response":[""]}

def create_graph() -> StateGraph:
    """Create and return the social worker graph"""
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("should_use_rag", should_use_rag)
    workflow.add_node("direct_response_social_worker", direct_response_social_worker)
    workflow.add_node("pass_state", pass_state)
    workflow.add_node("retrieve_and_generate", retrieve_and_generate)
    workflow.add_node("grader_docs", grader_docs)
    workflow.add_node("generate_context", generate_context)
    workflow.add_node("merge_responses", merge_responses)
    
    # Add edges
    workflow.add_edge(START, "should_use_rag")
    
    # Add conditional edges from RAG decision
    workflow.add_conditional_edges(
        "should_use_rag",
        lambda x: "use_rag" if x["should_rag"] else "direct_only",
        {
            "use_rag": "retrieve_and_generate",
            "direct_only": "pass_state"
        }
    )
    workflow.add_edge("should_use_rag", "direct_response_social_worker")
    # Add edge from direct response to merge
    workflow.add_edge("direct_response_social_worker", "merge_responses")
    workflow.add_edge("pass_state", "merge_responses")
    # Add edge from retrieve to grader
    workflow.add_conditional_edges("retrieve_and_generate", continue_to_grader,["grader_docs"])
    # Add edge from grader to generate context
    workflow.add_edge("grader_docs", "generate_context")
    # Add edge from generate context to merge
    workflow.add_edge("generate_context", "merge_responses")
    
    # Add final edge
    workflow.add_edge("merge_responses", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # Create the graph
    graph = create_graph()
    
    # Test questions
    test_cases = [
        {
            "question": "What resources are available for elderly care?",
            "domain": "financial",
            "conversation_history": ""
        },
        {
            "question": "what benefits do veterans have?",
            "domain": "financial",
            "conversation_history": ""
        },
        {
            "question": "I need help with mental health services. What's available?",
            "domain": "financial",
            "conversation_history": ""
        },
        {
            "question": "I'm feeling sad.",
            "domain": "financial",
            "conversation_history": ""
        },
    ]
    
    # Run test cases
    for case in test_cases:
        print(f"\nQuestion: {case['question']}")
        print(f"Domain: {case['domain']}")
        print("-" * 80)
        
        # try:
            # Initialize state
        state = AgentState(
            question=case["question"],
            conversation_history=case["conversation_history"],
            domain_filter=case["domain"],
            direct_response=None,
            rag_response=None,
            retrieved_context=None,
            retrieved_docs=None,
            final_response=None,
            should_rag=None,
            graded_docs=[]
        )
        # print(state)
        
        # Run the graph
        for output in graph.stream(state):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n---\n")
                
        # except Exception as e:
        #     print(f"Error processing question: {str(e)}")
        #     continue
graph = create_graph()