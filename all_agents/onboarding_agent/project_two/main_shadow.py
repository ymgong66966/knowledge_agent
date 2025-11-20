from google import genai
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Type, TypedDict, Annotated, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import random

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
import json

# test
memory = MemorySaver()


def get_gemini_api_key() -> str:
    """Get Gemini API key from OS environment or .env file"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
    return api_key


class CareRecipientInfo(BaseModel):
    # Optional: Pydantic v2 config if you need it
    model_config = ConfigDict(arbitrary_types_allowed=True)

    first_name: str
    last_name: str
    age: int
    gender: str


# ✅ Use TypedDict for LangGraph state (Pydantic 2 friendly)
class GraphState(BaseModel):

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    question: Optional[str] = Field(default=None)
    options: Optional[dict[str, str]] = Field(default=None)
    tasks: list[str] = Field(default= [])
    node: str = Field(default="root")
    user_response: Optional[str] = Field(default=None)
    chat_history: list[BaseMessage] = Field(default= [])
    # to_user: Optional[BaseMessage]
    next_step: Optional[str] = Field(default=None)
    real_chat_history: Optional[list[AnyMessage]] = Field(default=[])
    last_step: Optional[str] = Field(default=None)
    current_tree: str = Field(default="")
    route: Optional[str] = Field(default="onboarding")
    route_node: Optional[str] = Field(default="parse_response")
    mental_question: Optional[str] = Field(default="Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?")
    assessment_score: Optional[int] = Field(default=0)
    assessment_answer: Optional[list[AnyMessage]] = Field(default=[])
    care_recipient: Optional[dict] = Field(default={})
    completed_whole_process: Optional[bool] = Field(default=False)
    short_completed: Optional[bool] = Field(default=False)
    direct_record_answer: Optional[bool] = Field(default=False)
    directly_ask: Optional[bool] = Field(default=False)
    care_time: Optional[bool] = Field(default=False)
    veteranStatus: Optional[str] = Field(default="Not a veteran")


def root_node(state: GraphState) -> GraphState:
    return {
        "question": (
            "Hello! Welcome to WithCare. I’m your AI Care Navigator trained by "
            "licensed clinicians to support you 24/7. How are you today?"
        )
    }


graph = StateGraph(GraphState)
graph.add_node("root-node", root_node)
graph.add_edge(START, "root-node")
graph.add_edge("root-node", END)