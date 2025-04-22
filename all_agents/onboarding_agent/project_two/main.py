from google import genai
from pydantic import BaseModel, create_model
from typing import List, Optional, Type, TypedDict
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import random
from pydantic import BaseModel
from typing import TypedDict, Optional, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import Any
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict
from pydantic import Field
import json
from langchain_core.messages import AIMessage, AnyMessage
# test
memory = MemorySaver()

class carerecipient_info(BaseModel):
    first_name: str
    last_name: str
    age: int
    gender: str
    
class GraphState(BaseModel):
    question: Optional[str] = Field(default=None)
    options: Optional[dict[str, str]] = Field(default=None)
    tasks: list[str] = Field(default_factory=list)
    node: str = Field(default="root")
    user_response: Optional[str] = Field(default=None)
    chat_history: list[BaseMessage] = Field(default_factory=list)
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

class EndOfLifeCareNode:
    def __init__(self, question, options=None, tasks=None, condition=None, path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None  # Dictionary mapping answers to child nodes
        self.path = path
        self.tasks = tasks or None      # Tasks to add based on this node
        self.node_id = node_id        # Unique identifier for the node
        self.leaf_node = leaf_node or None
        self.next_questions = next_questions or None

class EndOfLifeCareTree:
    def __init__(self, person=None):
        # Node registry to store references to all nodes
        self.node_registry = {}
        self.person = person
        
        # Build the decision tree with node registration
        self._build_tree()
        self._build_next_questions()

    def _register_node(self, node, node_id):
        """Register a node with a unique ID for direct access"""
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node
    
    def get_node(self, node_id):
        """Get a node by its ID"""
        return self.node_registry.get(node_id)

    def _build_tree(self):
        # Leaf nodes (bottom of the tree)
        needs_support = self._register_node(EndOfLifeCareNode(
            question="We will add information on how to complete this to your Task list once the onboarding process is complete.",
            tasks=["Find End of Life Planning Support"],
            leaf_node="leaf"
        ), "needs_support")
        
        no_support_needed = self._register_node(EndOfLifeCareNode(
            question="",  # Terminal node with no further questions
            tasks=[],
            leaf_node="leaf"
        ), "no_support_needed")
        
        # Root node
        self.root = self._register_node(EndOfLifeCareNode(
            question="Is @name in need of end-of-life care or planning support?",
            options={
                "Yes": "needs_support",
                "No": "no_support_needed"
            }
        ), "root")
    def _build_next_questions(self):
        for node_id in self.node_registry.keys():
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                print(node.options)
                for option in node.options.values():
                    print(option)
                    option_node = self.get_node(option)
                    print(option_node, option)
                    while option_node.options is None and option_node.leaf_node != "leaf" and option_node.path is not None:
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions
                self.node_registry[node_id] = node

class LegalDocumentsNode:
    def __init__(self, question, options=None, tasks=None, condition=None, path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None  # Dictionary mapping answers to child nodes
        self.path = path
        self.tasks = tasks or None      # Tasks to add based on this node
        self.node_id = node_id        # Unique identifier for the node
        self.leaf_node = leaf_node or None
        self.next_questions = next_questions or None

class LegalDocumentsTree:
    def __init__(self, person=None):
        # Node registry to store references to all nodes
        self.node_registry = {}
        self.person = person
        
        # Build the decision tree with node registration
        self._build_tree()
        self._build_next_questions()

    def _register_node(self, node, node_id):
        """Register a node with a unique ID for direct access"""
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node
    
    def get_node(self, node_id):
        """Get a node by its ID"""
        return self.node_registry.get(node_id)

    def _build_tree(self):
        # Leaf nodes (bottom of the tree)
        has_documents = self._register_node(LegalDocumentsNode(
            question="Great. Make sure you have shared these documents with relevant parties including medical and financial institutions. We will add this to your Task list once the onboarding process is complete.",
            tasks=["Gather Important Documents"],
            leaf_node="leaf"
        ), "has_documents")
        
        not_sure = self._register_node(LegalDocumentsNode(
            question="It will be important to check with @name to see if these legal plans have been completed. We will add this to your Task list once the onboarding process is complete.",
            tasks=["Gather Important Documents"],
            leaf_node="leaf"
        ), "not_sure")
        
        no_documents = self._register_node(LegalDocumentsNode(
            question="It will be important to work with @name and an attorney to complete these documents as soon as possible. We will add this to your Task list with more instructions once the onboarding process is complete.",
            tasks=["Find an Attorney", "Create a Living Will", "Complete a Healthcare Power of Attorney", "Establish Power of Attorney"],
            leaf_node="leaf"
        ), "no_documents")
        
        # Root node
        self.root = self._register_node(LegalDocumentsNode(
            question="Does @name have a will, living will, or power of attorney in place?",
            options={
                "Yes": "has_documents",
                "I'm not sure": "not_sure",
                "No": "no_documents"
            }
        ), "root")
    def _build_next_questions(self):
        for node_id in self.node_registry.keys():
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                for option in node.options.values():
                    option_node = self.get_node(option)
                    while option_node.options is None and option_node.leaf_node != "leaf" and option_node.path is not None:
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions
                self.node_registry[node_id] = node


class MedicareAssessmentNode:
    def __init__(self, question, options=None, tasks=None, condition=None, path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None  # Dictionary mapping answers to child nodes
        self.path = path
        self.tasks = tasks or None      # Tasks to add based on this node
        self.node_id = node_id        # Unique identifier for the node
        self.leaf_node = leaf_node or None
        self.next_questions = next_questions or None

class MedicareAssessmentTree:
    def __init__(self, person=None):
        # Node registry to store references to all nodes
        self.node_registry = {}
        self.person = person
        
        # Build the decision tree with node registration
        self._build_tree()
        self._build_next_questions()
    
    def _register_node(self, node, node_id):
        """Register a node with a unique ID for direct access"""
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node
    
    def get_node(self, node_id):
        """Get a node by its ID"""
        return self.node_registry.get(node_id)

    def _build_tree(self):
        # Leaf nodes (bottom of the tree)
        already_enrolled = self._register_node(MedicareAssessmentNode(
            question="",  # Terminal node with no further questions
            tasks=["Explore Medicare Benefits"], 
            leaf_node="leaf"
        ), "already_enrolled")
        
        not_enrolled = self._register_node(MedicareAssessmentNode(
            question="Information on how to enroll in Medicare will be provided in @name's Task list once the onboarding process is complete",
            tasks=["Enroll in Medicare"],
            leaf_node="leaf"
        ), "not_enrolled")
        
        not_eligible = self._register_node(MedicareAssessmentNode(
            question="Let's explore what other benefits or resources @name might be eligible for.",
            tasks=[],
            leaf_node="leaf"
        ), "not_eligible")
        
        # Middle-level nodes
        is_enrolled = self._register_node(MedicareAssessmentNode(
            question="Great. Is @name already enrolled in Medicare?",
            options={
                "Yes": "already_enrolled",
                "No": "not_enrolled"
            }
        ), "is_enrolled")
        
        # Root node
        self.root = self._register_node(MedicareAssessmentNode(
            question="Is @name eligible for Medicare? Medicare is a federal health insurance program for those aged 65 or older, or who have a qualifying disability or diagnosis of end-stage kidney disease or ALS.",
            options={
                "Yes": "is_enrolled",
                "No": "not_eligible"
            }
        ), "root")

    def _build_next_questions(self):
        for node_id in self.node_registry.keys():
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                for option in node.options.values():
                    option_node = self.get_node(option)
                    while option_node.options is None and option_node.leaf_node != "leaf" and option_node.path is not None:
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions
                self.node_registry[node_id] = node

class VeteranAssessmentNode:
    def __init__(self, question, options=None, tasks=None, condition=None, path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None  # Dictionary mapping answers to child nodes
        self.path = path
        self.tasks = tasks or None      # Tasks to add based on this node
        self.node_id = node_id        # Unique identifier for the node
        self.leaf_node = leaf_node or None
        self.next_questions = next_questions or None

class VeteranAssessmentTree:
    def __init__(self, person=None):
        # Node registry to store references to all nodes
        self.node_registry = {}
        self.person = person
        
        # Build the decision tree with node registration
        self._build_tree()
        self._build_next_questions()
    
    def _register_node(self, node, node_id):
        """Register a node with a unique ID for direct access"""
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node
    
    def get_node(self, node_id):
        """Get a node by its ID"""
        return self.node_registry.get(node_id)
    
    def _build_tree(self):
        # Leaf nodes (bottom of the tree)
        accessing_benefits = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="Good to know that @name is currently accessing veterans benefits.",  # Terminal node with no further questions
                tasks=[],  # No additional tasks needed if already accessing benefits
                leaf_node="leaf"
            ),
            "accessing_benefits"
        )
        
        not_accessing_benefits = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="Information on how to explore eligibility for VA benefits will be added to your Task list once the onboarding process is complete. ",
                tasks=["Evaluate Veteran VA Benefits"],
                leaf_node="leaf"
            ),
            "not_accessing_benefits"
        )
        
        not_veteran = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="",  # Terminal node with no further questions
                tasks=[],
                leaf_node="leaf"
            ),
            "not_veteran"
        )
        
        # Middle-level nodes
        is_accessing_benefits = self._register_node(
            VeteranAssessmentNode(
                question="Is @name currently accessing veterans benefits?",
                options={
                    "Yes": "accessing_benefits",
                    "No": "not_accessing_benefits"
                },
                tasks=[]
            ),
            "is_accessing_benefits"
        )
        
        # Root node
        self.root = self._register_node(
            VeteranAssessmentNode(
                question="Is @name a veteran?",
                options={
                    "Yes": "is_accessing_benefits",
                    "No": "not_veteran"
                },
                tasks=[]
            ),
            "root"
        )

    def _build_next_questions(self):
        for node_id in self.node_registry.keys():
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                for option in node.options.values():
                    option_node = self.get_node(option)
                    while option_node.options is None and option_node.leaf_node != "leaf" and option_node.path is not None:
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions
                self.node_registry[node_id] = node

class HousingAssessmentNode:
    def __init__(self, question, options=None, tasks=None, condition=None, path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None  # Dictionary mapping answers to child nodes
        self.path = path
        self.tasks = tasks or None      # Tasks to add based on this node
        # self.condition = condition    # Optional condition function (e.g., age check)
        self.node_id = node_id        # Unique identifier for the node
        self.leaf_node = leaf_node or None
        self.next_questions = next_questions or None

class HousingAssessmentTree:
    def __init__(self, person=None):
        # Node registry to store references to all nodes
        self.node_registry = {}
        self.person = person
        
        # Build the decision tree with node registration
        self._build_tree()
        self._build_next_questions()
    
    def _register_node(self, node, node_id):
        """Register a node with a unique ID for direct access"""
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node
    
    def get_node(self, node_id):
        """Get a node by its ID"""
        return self.node_registry.get(node_id)
    
    def _build_tree(self):
        # Leaf nodes (bottom of the tree)
        need_assistance_yes = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="We will provide information on how to hire in-home care in your Tasks list once the onboarding process is complete.",
                tasks=["Find In-Home Support"],
                leaf_node="leaf"
            ), 
            "need_assistance_yes"
        )
        
        need_assistance_no = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="",  # Terminal node with no further questions
                tasks=[],
                leaf_node="leaf"
            ),
            "need_assistance_no"
        )
        
        # Middle-level nodes
        need_assistance = self._register_node(
            HousingAssessmentNode(
                
                question="Does @name need assistance with activities of daily living such as dressing, bathing or preparing meals?",
                options={
                    "Yes": "need_assistance_yes",
                    "No": "need_assistance_no"
                },
                tasks=[]
            ),
            "need_assistance"
        )
        
        need_more_support_yes = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="We will provide information on how to hire in-home care in your Tasks list once the onboarding process is complete.",
                tasks=["Find In-Home Support"],
                leaf_node="leaf"
            ),
            "need_more_support_yes"
        )
        
        need_more_support_not_sure = self._register_node(
            HousingAssessmentNode(
                question="Does @name need assistance with activities of daily living such as dressing, bathing or preparing meals?",
                options={
                    "Yes": "need_assistance_yes",
                    "No": "need_assistance_no"
                },
                tasks=[]
            ),
            "need_more_support_not_sure"
        )
        
        need_more_support = self._register_node(
            HousingAssessmentNode(
                question="Does @name need more support or care at home?",
                options={
                    "Yes": "need_more_support_yes",
                    "No": "need_more_support_not_sure"
                },
                tasks=[]
            ),
            "need_more_support"
        )
        
        receive_care_yes = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="",  # No additional question
                tasks=["Monitor In-Home or In-Facility Care"]
            ),
            "receive_care_yes"
        )
        
        receive_care = self._register_node(
            HousingAssessmentNode(
                question="Does @name receive in-home care or assistance?",
                options={
                    "Yes": "receive_care_yes",
                    "No": "need_more_support"
                },
                tasks=[]
            ),
            "receive_care"
        )
        
        # Second-level nodes
        lives_alone = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="",  # This will be determined by the next question
                path="receive_care",  # Default next node
                tasks=["Explore Technology for Older Adults"]
                # This adds a task if the person is over 65
            ),
            "lives_alone"
        )
        
        need_more_support_with_others = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="We will provide information on how to hire in-home care in your Tasks list once the onboarding process is complete.",
                tasks=["Find In-Home Support"],
                leaf_node="leaf"
            ),
            "need_more_support_with_others"
        )
        
        no_support_with_others = self._register_node(
            HousingAssessmentNode(
                options=None,
                question="",  # No additional question
                tasks=[],
                leaf_node="leaf"
            ),
            "no_support_with_others"
        )
        
        not_sure_with_others = self._register_node(
            HousingAssessmentNode(

                question="Does @name need assistance with activities of daily living such as dressing, bathing or preparing meals?",
                options={"Yes": "need_more_support_with_others", "No": "no_support_with_others"},
                tasks=[]
            ),
            "not_sure_with_others"
        )

        lives_with_others = self._register_node(
            HousingAssessmentNode(
                question="Great. Does @name need more support or care than is currently being provided by those he/she lives with?",  # No specific question for this path
                options={"Yes": "need_more_support_with_others", "No": "not_sure_with_others"},
                tasks=[]
            ),
            "lives_with_others"
        )

        lives_with = self._register_node(
            HousingAssessmentNode(
                question="Does @name live in this home alone or with others?",
                options={
                    "Alone": "lives_alone",
                    "With others": "lives_with_others"
                },
                tasks=[]
            ),
            "lives_with"
        )
        
        # Root node
        self.root = self._register_node(
            HousingAssessmentNode(
                question="Does @name live in a private residence home or in a care facility?",
                options={
                    "A private residence / single family home / apartment / independent living facility": "lives_with",
                    "A care facility / assisted living / memory care / board and care / skilled nursing facility": "receive_care_yes"
                },
                tasks=[]
            ),
            "root"
        )

    def _build_next_questions(self):
        for node_id in self.node_registry.keys():
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                for option in node.options.values():
                    option_node = self.get_node(option)
                    while option_node.options is None and option_node.leaf_node != "leaf" and option_node.path is not None:
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions
                self.node_registry[node_id] = node



class to_be_parsed_model(BaseModel):
    option: str = Field(None)
    has_additional_info: str = Field(None)

def assess_mental(state: GraphState, question_ls: list):
    client = genai.Client(api_key="AIzaSyCGUPJfjJdIn8vu78HD8wq9j-zdbWRy2mk")
    user_response = ""
    for message in reversed(state.real_chat_history):
        if isinstance(message, HumanMessage):
            user_response = message.content + user_response
        elif isinstance(message, AIMessage):
            break
    mental_question = state.mental_question
    prompt = f"""System: you are doing mental assessment for the user. Your input contains the user response and the assessment question. You need to return the option that best represents the user's response. Your output should be a json object with one field: option. The option should be one of these three options: \\"yes\\", \\"no\\", \\"sometimes\\".
    Here is an example:

    Input: 
    assessment question: "Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?"
    user response: "Not really"
    Output: {{"option": "no"}}

    #########
    Here is the actual input:
    assessment question: {mental_question}
    user response: {user_response}
    Make sure your output is in json format. The key is "option". The value should be one of these three options: "yes", "no", "sometimes".
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
 
        },
    )
    response_json = json.loads(response.text)
    assessment_score = state.assessment_score
    assessment_answer = state.assessment_answer
    assessment_answer += [AIMessage(content=mental_question), HumanMessage(content=response_json.get("option", "no"))]
    if response_json.get("option", "no").lower() == "yes":
        assessment_score += 2
    elif response_json.get("option", "no").lower() == "sometimes":
        assessment_score += 1
    else:
        assessment_score += 0
    if question_ls.index(mental_question) == len(question_ls) - 1:
        return {"next_step": "completed_whole", "assessment_score": int(assessment_score), "assessment_answer": assessment_answer,  "real_chat_history": state.real_chat_history + [HumanMessage(content=response_json.get("option", "no"))], "chat_history": state.chat_history + [HumanMessage(content=response_json.get("option", "no"))]}
    else:
        mental_question_new = question_ls[question_ls.index(mental_question) + 1]
        return {"next_step": END, "assessment_score": int(assessment_score), "assessment_answer": assessment_answer, "mental_question": mental_question_new, "real_chat_history": state.real_chat_history + [HumanMessage(content=response_json.get("option", "no")), AIMessage(content=mental_question_new)], "chat_history": state.chat_history + [HumanMessage(content=response_json.get("option", "no")), AIMessage(content=mental_question_new)], "question": mental_question_new}


def routing_node(state: GraphState):
    if state.route == "mental":
        return {"route_node": "assess_mental"}
    else:
        return {"route_node": "parse_response"}

def parse_response(state: GraphState, tree_dict: dict):
    # print(state['node'].options)
    # return {"options": state['node'].options, "node":list(state['node'].options.values())[0]}
    # print(state["node"], "////////")
    tree = tree_dict[state.current_tree]
    client = genai.Client(api_key="AIzaSyCGUPJfjJdIn8vu78HD8wq9j-zdbWRy2mk")
    current_question = tree.get_node(state.node).question
    # print("Next Questions:", tree.get_node(state.node).next_questions)
    next_questions = tree.get_node(state.node).next_questions
    all_questions = "Current Question: " + current_question + "\nLater Questions: " + str(next_questions)

    user_response = ""
    for message in reversed(state.real_chat_history):
        if isinstance(message, HumanMessage):
            user_response = message.content + user_response
        elif isinstance(message, AIMessage):
            break
    tasks = state.tasks

    chat_history = state.chat_history
    # print(tree.get_node(state.node).options)
    if tree.get_node(state.node).options:
        options = list(tree.get_node(state.node).options.keys())
    else:
        options = [tree.get_node(state.node).path]
    # print("options", options)


    prompt = f"""System: we have a series of questions to ask the user to help them onboard. Your input contains four things: user response, current question, options, and later questions. Your job is to understand the user's response to your current question, and return one of the options you are given that best represent the user's response. At the same time, you need to look at the later questions and return a "has_additional_info" flag indicating if there is additional information in the user's response that can answer later questions. The "has_additional_info" is either "True" or "False". If the user's response has information to answer one of the later questions, then you must return "True" as the value for "has_additional_info". If the user's response has no information to answer any of the later questions, then you must return "False" as the value for "has_additional_info". 

    Note that there are three scenarios:
    1. The user's response is sufficient for you to pick one option as answer to the current question.

    Here is an example:
    current question: "Does @name live in this home alone or with others?"
    later questions: ["Does @name need additional assistance?"]
    user response: "he is living with others"
    options: ["with others", "alone"]

    Output: {{"option": "with others", "has_additional_info": "False"}}
    
    Here is another example:

    current question: "Does @name need additional assistance?"
    later questions: ["Does @name live in this home alone or with others?"]
    user response: "No he doesn't."
    options: ["Yes", "No"]

    Output: {{"option": "No", "has_additional_info": "False"}}
    Because "No he doesn't" was meant to answer the current question. There is no information left in the response that can answer any more questions.

    2. The user's response is insufficient for you to pick one option as answer to the current question.

    Here is an example:
    current question: "Does @name live in this home alone or with others?"
    later questions: ["how old is @name?"]
    user response: "I'm not sure"
    options: ["with others", "alone"]

    Output: {{"option": "answer not found", "has_additional_info": "False"}}
    Because "I'm not sure" was not enough to answer the current question. There is no information left in the response that can answer any more questions.

    Here is another example:
    current question: "Great. Does @name need more support or care than is currently being provided by those he/she lives with?"
    later questions: ["how old is @name?"]
    user response: "he is living with my family, and he is 70 years old"
    options: ["yes", "no"]

    Output: {{"option": "answer not found", "has_additional_info": "True"}}
    This is because the user's response is not clear on whether he needs more support or not.

    Here is another example:
    current question: "Great. Does @name need more support or care than is currently being provided by those he/she lives with?
    user response: "yes he is doing that, and he is 70 years old"
    options: ["yes", "no"]

    Output: {{"option": "answer not found", "has_additional_info": "True"}}
    Although there is a yes, the user's response content is not related to the question.

    3. The user's response is sufficient for you to pick one option as answer to the current question, and there is additional information that can answer later questions.

    Here is an example:
    current question: "Does @name live in this home alone or with others?"
    later questions: ['Does @name receive in-home care or assistance?', 'Great. Does @name need more support or care than is currently being provided by those he/she lives with?']
    user response: "he is living with my family, but we don't have time or energy to take care of him"
    options: ["with others", "alone"]

    Output: {{"option": "with others", "has_additional_info": "True"}}

    Here is another example:
    current question: "Does @name live in this home alone or with others?"
    later questions: ['Does @name receive in-home care or assistance?', 'Great. Does @name need more support or care than is currently being provided by those he/she lives with?']
    user response: "He is living alone, but we take good care of him and visit him often"
    options: ["with others", "alone"]

    Output: {{"option": "alone", "has_additional_info": "True"}}


    ### Below is the real current question, real user response, and real current options. 
    {all_questions}
    user response: {user_response}
    options: {options}

    Generate your output in the json format below, do not include any other text.

    {{"option": "", "has_additional_info": ""}}

    Important: as long as the user's reponse is only expressing a postive or negative sentiment, no matter how lengthy they language is, we think there is no additional information in it.
    Important, I need you to be very strict with labeling the "has_additional_info" field. A building will be set on fire if you mistakingly label a "has_additional_info" as True when it is actually False. 
    Important: you mush first check if the user's response can answer the question, if not, you must return "answer not found" as the option. 
    """
    

    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
 
        },
    )

    # three scenarios: 1. {option: not none, has_additional_info: True}
    # move to the next node, try to answer the next question
    # 2. {option: not none, has_additional_info: False}
    # move to the next node, not try to answer the next question
    # 3. {option: none, has_additional_info: True/False}
    # route to the ask_to_repeat
    
    response_dict = json.loads(response.text)
    real_chat_history = state.real_chat_history
    # print("response_dict", response_dict)
    if response_dict["option"] != "answer not found":
        parsed_option = response_dict["option"]
        has_additional_info = response_dict["has_additional_info"]
        chat_history += [AIMessage(content=current_question),HumanMessage(content=parsed_option)]
        real_chat_history += [AIMessage(content=current_question),HumanMessage(content=user_response)]
        # update the current node
        node_id = tree.get_node(state.node).options[parsed_option]
        current_node = tree.get_node(node_id)
        # print("current_node", node_id)
        # print("#####",current_node.options)
        while current_node.options is None:
            if current_node.tasks != None:
                # print(current_node.tasks, "tasks")
                tasks += current_node.tasks
            path = current_node.path
            # print("path", path)
            if path is None:
                break
            else:
                current_node = tree.get_node(path)
                node_id = path 
        if current_node.leaf_node == "leaf":
            if current_node.question:
                current_question = current_node.question
            else: 
                current_question = ""
            idx = list(tree_dict.keys()).index(state.current_tree) + 1 
            if idx < len(tree_dict):
                current_tree_name = list(tree_dict.keys())[idx]
                return {"current_tree": current_tree_name, "next_step": "ask_next_question", "last_step": "start", "node": "root", "question": current_question,"chat_history": chat_history, "real_chat_history":real_chat_history}
            else:

                return {"next_step": "completed_onboarding", "question": current_question,"chat_history": chat_history, "real_chat_history":real_chat_history}

        current_question = current_node.question
        options = current_node.options
        if has_additional_info.lower() == "true":
            return {"node": node_id, "chat_history": chat_history, "tasks": tasks, "options": options, "question":current_question, "next_step": "parse_response", "last_step": "parse_response", "real_chat_history":real_chat_history}
        else:
            return {"node": node_id, "chat_history": chat_history, "tasks": tasks, "options": options, "question":current_question, "next_step": "ask_next_question", "last_step": "parse_response", "real_chat_history":real_chat_history}
        # the node can be any node in the tree. Need to check if the node has options
    # if: 1. the node has options, then it must has question. then options = node.options
    # else: 2. the node has no options, then it must has path. then options = node.path
    # if there is a task list associated with the node, append it to the task field
    elif response_dict["option"] == "answer not found" and state.last_step == "parse_response":
        return {"next_step": "ask_next_question", "last_step": "start"}
    elif response_dict["option"] == "answer not found" and state.last_step != "parse_response":
        return {"next_step": "ask_to_repeat", "last_step": "start"}

def ask_to_repeat(state: GraphState):
    new_history = state.real_chat_history + [AIMessage(content="I'm sorry, I didn't understand that. Please try again.")]
    return {"real_chat_history": new_history}
        
def completed_onboarding(state: GraphState):
    new_history = state.real_chat_history + [AIMessage(content="Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?")]
    return {"real_chat_history": new_history, "question": "Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?", "chat_history": state.chat_history + [AIMessage(content="Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?")], "route": "mental"}

async def completed_whole(state: GraphState):
    assess_score = state.assessment_score
    if assess_score < 5:
        new_history = state.real_chat_history + [AIMessage(content="You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and use the Care Navigator to find resources you may need.")]
        question = "You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and use the Care Navigator to find resources you may need."
        chat_history = state.chat_history + [AIMessage(content="You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and use the Care Navigator to find resources you may need.")]
    elif assess_score >= 5 and assess_score < 11:
        new_history = state.real_chat_history + [AIMessage(content="Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving.")]
        question = "Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving."
        chat_history = state.chat_history + [AIMessage(content="Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving.")]
    elif assess_score >= 11:
        new_history = state.real_chat_history + [AIMessage(content="Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask your Care Navigator for assistance with finding additional resources such as support groups or individual therapists.")]
        question = "Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask your Care Navigator for assistance with finding additional resources such as support groups or individual therapists."
        chat_history = state.chat_history + [AIMessage(content="Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask your Care Navigator for assistance with finding additional resources such as support groups or individual therapists.")]

    return {"real_chat_history": new_history, "question":question, "chat_history": chat_history}

def ask_next_question(state: GraphState, tree_dict: dict):
    # print(state["real_chat_history"], "\\\\\\\\\\")
    client = genai.Client(api_key="AIzaSyCGUPJfjJdIn8vu78HD8wq9j-zdbWRy2mk")
    care_recipient = str(state.care_recipient)
    # current_question = "Is there any immediate need I can support you with right now in regards to @name's care?"
    # chat_history = [AIMessage(content="Is your dad eligible for Medicaid? Medicaid is a government program that provides healthcare coverage for low-income individuals.", role="assistant"), HumanMessage(content="Yes, he is", role="user")]
    # chat_history = []
    if state.node == "root":
        current_question = state.question + " " + tree_dict[state.current_tree].get_node(state.node).question
    else:
        current_question = tree_dict[state.current_tree].get_node(state.node).question
    real_chat_history = state.real_chat_history
    ask_template = f"""System: your job is to ask the question to the care giver to help them onboard. Your input includes a question, the chat history, and the care recipient information. 

    You need to undertstand the relationship between the care giver you are talking to and the care recipient. Refer to the care recipient based on the relationship. If the relationship is "self" then refer to the care recipient as you. Also look at the background information in chat history, formulate the question accordingly.

    Here are some example inputs:
    The care recipient information:
    {{
  "address": "11650 National Boulevard, Los Angeles, California 90064, United States",
  "dateOfBirth": "1954-04-11",
  "dependentStatus": "Not a child/dependent",
  "firstName": "Justin",
  "gender": "Male",
  "isSelf": false,
  "lastName": "Timberlake",
  "legalName": "Justin Timberlake",
  "pronouns": "he/him/his",
  "relationship": "Brother",
  "veteranStatus": "Veteran"
}}
    Input question: Great. Does @name need more support or care than is currently being provided by those he/she lives with?
    Chat history: [AIMessage(content="Does your brother live in this home alone or with others?", role="assistant"), HumanMessage(content="He is living with my family", role="user")]

    The example output:
    The output question you need to ask is: Does your brother need more support or care than is currently being provided by your family?

    #### Below is the real input. 
    The care recipient information:
    {care_recipient}
    The question is {current_question}.  
    Chat history: {real_chat_history}
    
    Generate your output in the json format below, do not include any other text.
    
    {{"question": ""}}
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=ask_template,
        config={
            'response_mime_type': 'application/json',
 
        },
    )

    # print(response.text)
    generated_question = json.loads(response.text).get("question")
    real_chat_history = state.real_chat_history
    real_chat_history.append(AIMessage(content=generated_question))
    return {"question": generated_question, "real_chat_history": real_chat_history, "last_step":"start"}

def create_graph():
    memory = MemorySaver()
    builder = StateGraph(GraphState)
    tree_dict = {"HousingAssessmentTree":HousingAssessmentTree(), "Medic" "VeteranAssessmentTree":VeteranAssessmentTree(), "MedicareAssessmentTree":MedicareAssessmentTree(), "LegalDocumentsTree":LegalDocumentsTree(), "EndOfLifeCareTree":EndOfLifeCareTree()}
    mental_health_questions_ls = ["Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?", "Have you been sleeping more or less often than usual? Yes, no, or sometimes?", "Do you feel lonely or isolated? Yes, no, or sometimes?", "Have you lost interest in activities that you used to enjoy? Yes, no, or sometimes?", "Do you feel anxious, or like you can’t stop worrying about things that might happen? Yes, no, or sometimes?", "Do you feel down, sad or depressed? Yes, no, or sometimes?"]
    care_recipient = {"address": "11650 National Boulevard, Los Angeles, California 90064, United States",
    "dateOfBirth": "1954-04-11",
    "dependentStatus": "Not a child/dependent",
    "firstName": "Yiming",
    "gender": "Male",
    "isSelf": False,
    "lastName": "Gong",
    "legalName": "Yiming Gong",
    "pronouns": "he/him/his",
    "relationship": "dad",
    "veteranStatus": "Veteran"}
    builder.add_node("parse_response", lambda state: parse_response(state, tree_dict))
    builder.add_node("ask_to_repeat", ask_to_repeat)
    builder.add_node("ask_next_question", lambda state: ask_next_question(state, tree_dict))
    builder.add_node("assess_mental", lambda state: assess_mental(state, mental_health_questions_ls))
    builder.add_node("router_node", routing_node)
    builder.add_node("completed_onboarding", completed_onboarding)
    builder.add_node("completed_whole", completed_whole)
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("parse_response", lambda x: x.next_step, {"ask_to_repeat": "ask_to_repeat", "ask_next_question": "ask_next_question", "parse_response": "parse_response", "completed_onboarding": "completed_onboarding"})
    builder.add_conditional_edges("router_node", lambda x: x.route_node, {"assess_mental": "assess_mental", "parse_response": "parse_response"})
    builder.add_conditional_edges("assess_mental", lambda x: x.next_step, {END: END, "completed_whole": "completed_whole"})
    builder.add_edge("completed_whole", END)
    builder.add_edge("completed_onboarding", END)

    builder.add_edge("ask_to_repeat", END)
    builder.add_edge("ask_next_question", END)
    graph = builder.compile(checkpointer=memory)

    return graph

graph = create_graph()

# if __name__ == "__main__":
#     memory = MemorySaver()
#     builder = StateGraph(GraphState)
#     tree_dict = {"HousingAssessmentTree":HousingAssessmentTree(), "VeteranAssessmentTree":VeteranAssessmentTree()}
#     mental_health_questions_ls = ["Next I am going to ask you some questions about how you have been managing in your role as a care provider. Do you have trouble concentrating? Yes, no, or sometimes?", "Have you been sleeping more or less often than usual? Yes, no, or sometimes?", "Do you feel lonely or isolated? Yes, no, or sometimes?", "Have you lost interest in activities that you used to enjoy? Yes, no, or sometimes?", "Do you feel anxious, or like you can’t stop worrying about things that might happen? Yes, no, or sometimes?", "Do you feel down, sad or depressed? Yes, no, or sometimes?"]
#     care_recipient = {"address": "11650 National Boulevard, Los Angeles, California 90064, United States",
#   "dateOfBirth": "1954-04-11",
#   "dependentStatus": "Not a child/dependent",
#   "firstName": "Yiming",
#   "gender": "Male",
#   "isSelf": False,
#   "lastName": "Gong",
#   "legalName": "Yiming Gong",
#   "pronouns": "he/him/his",
#   "relationship": "dad",
#   "veteranStatus": "Veteran"}
#     builder.add_node("parse_response", lambda state: parse_response(state, tree_dict))
#     builder.add_node("ask_to_repeat", ask_to_repeat)
#     builder.add_node("ask_next_question", lambda state: ask_next_question(state, tree_dict))
#     builder.add_node("assess_mental", lambda state: assess_mental(state, mental_health_questions_ls))
#     builder.add_node("router_node", routing_node)
#     builder.add_node("completed_onboarding", completed_onboarding)
#     builder.add_node("completed_whole", completed_whole)
#     builder.add_edge(START, "router_node")
#     builder.add_conditional_edges("parse_response", lambda x: x.next_step, {"ask_to_repeat": "ask_to_repeat", "ask_next_question": "ask_next_question", "parse_response": "parse_response", "completed_onboarding": "completed_onboarding"})
#     builder.add_conditional_edges("router_node", lambda x: x.route_node, {"assess_mental": "assess_mental", "parse_response": "parse_response"})
#     builder.add_conditional_edges("assess_mental", lambda x: x.next_step, {END: END, "completed_whole": "completed_whole"})
#     builder.add_edge("completed_whole", END)
#     builder.add_edge("completed_onboarding", "assess_mental")

#     builder.add_edge("ask_to_repeat", END)
#     builder.add_edge("ask_next_question", END)
#     graph = builder.compile(checkpointer=memory)

#     config = {"configurable": {"thread_id": "3"}}
#     result = graph.invoke({"node": "root", "tasks": [], "chat_history": [], "real_chat_history": [HumanMessage(content="He is living in an apartment, alone by himself")], "last_step": "start", "current_tree": "HousingAssessmentTree", "care_recipient": care_recipient}, config=config)

#     state = graph.get_state(config)
#     print(result["question"])
#     # print(state.values["real_chat_history"])
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="No he does not")]})
#     # print(graph.get_state(branch_config).values, "......")
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print(result["question"])
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="No he does not")]})
#     # print(graph.get_state(branch_config).values, "......")
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print(result["question"])
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="No he does not need")]})
#     # print(graph.get_state(branch_config).values, "......")
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print(result["question"])

#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes, he is a veteran. He is accessing veteran benefits")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print(result["question"])
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print("......",result["question"])
#     state = graph.get_state(config)
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print("......",result["question"])
#     state = graph.get_state(config)
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print("......",result["question"])
#     state = graph.get_state(config)
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print("......",result["question"])
#     state = graph.get_state(config)
#     branch_config = graph.update_state(config, {"real_chat_history": state.values["real_chat_history"] + [HumanMessage(content="Yes")]})
#     result = graph.invoke(graph.get_state(branch_config).values, config=branch_config)
#     print("......",result["question"])
#     print(result)

