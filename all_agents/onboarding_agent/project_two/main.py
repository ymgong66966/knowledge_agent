# from google import genai
# # from pydantic import BaseModel, create_model
# # from langchain_core.pydantic_v1 import BaseModel, Field
# from pydantic import BaseModel, Field, ConfigDict
# from typing import List, Optional, Type, TypedDict
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from openai import OpenAI
# import random
# # from pydantic import BaseModel
# from typing import TypedDict, Optional, Annotated, List
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# from typing import Any
# from langgraph.checkpoint.memory import MemorySaver
# import json
# from langchain_core.messages import AIMessage, AnyMessage

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

def get_gemini_api_key():
    """Get Gemini API key from OS environment or .env file"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Load from .env if not in OS environment
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
    return api_key

class carerecipient_info(BaseModel):
    first_name: str
    last_name: str
    age: int
    gender: str
    
class GraphState(BaseModel):

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    question: Optional[str] = Field(default=None)
    options: Optional[dict[str, str]] = Field(default=None)
    tasks: list[str] = Field(default= [])
    node: str = Field(default="root")
    user_response: Optional[str] = Field(default=None)
    chat_history: list= Field(default= [])
    # to_user: Optional[BaseMessage]
    next_step: Optional[str] = Field(default=None)
    real_chat_history: Optional[list] = Field(default=[])
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

class IntroNode:
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None       # Dict[str, str]: choice -> child node_id
        self.tasks = tasks or None           # List[str]
        self.condition = condition           # Optional callable/flag
        self.path = path                     # Optional linear path pointer (auto-advance)
        self.node_id = node_id               # Unique identifier
        self.leaf_node = leaf_node or None   # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Filled after build


class IntroAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # Registry helpers
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # Build tree
    def _build_tree(self):
        # Leaves
        leaf_positive_ack = self._register_node(IntroNode(
            question="Glad to hear that! I hope our conversation today will be helpful for you.\n\nBefore we continue, is there anything specific about @name’s care you need help figuring out or getting started on today?",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_positive_ack")

        leaf_negative_ack = self._register_node(IntroNode(
            question="I’m really sorry to hear that. I’ll do my best to make things a little lighter for you today.\n\nBefore we continue, is there anything specific about @name’s care you need help figuring out or getting started on today?",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_negative_ack")

        # Root (frontend also shows this intro; we keep it here for completeness)
        self.root = self._register_node(IntroNode(
            question=("Hello! Welcome to WithCare. I’m your AI Care Navigator trained by licensed clinicians to support you 24/7. How are you today?"),
            options={
                "The tone of user's answer is Positive/Neutral": "leaf_positive_ack",
                "The tone of user's answer is Negative": "leaf_negative_ack"
            }
        ), "root")

    # Precompute next_questions (same pattern as your other trees)
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    while (option_node is not None and
                           option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)

                    if option_node is None or option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)

                node.next_questions = next_questions

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
            tasks=["End of Life Planning Support"],
            leaf_node="leaf"
        ), "needs_support")
        
        no_support_needed = self._register_node(EndOfLifeCareNode(
            question="""That’s okay. You are not required to answer anything you do not feel comfortable with. We can move forward to the next question.
""",  # Terminal node with no further questions
            tasks=[],
            leaf_node="leaf"
        ), "no_support_needed")
        
        # Root node
        self.root = self._register_node(EndOfLifeCareNode(
            question="""Depending on @name’s diagnosis, it may be important to consider end-of-life planning so that they can remain comfortable with their wishes honored. \n\nEnd-of-life care can include hospice or palliative care, spiritual care, healthcare directive and conversations with family, medical providers, and even attorneys. \n\nIs @name in need of end-of-life care or planning support?""",
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
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None       # Dict[str, str] mapping choice -> child node_id
        self.tasks = tasks or None           # List[str] tasks at this node
        self.condition = condition           # Optional callable/flag for conditional routing
        self.path = path                     # Optional linear path pointer (auto-advance)
        self.node_id = node_id               # Unique identifier
        self.leaf_node = leaf_node or None   # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Filled after build


class LegalDocumentsAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # ----------------------------
    # Registry helpers
    # ----------------------------
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # ----------------------------
    # Tree construction (a → b → c → d)
    # ----------------------------
    def _build_tree(self):
        # Task labels
        TASK_FIND_ATTORNEY = ["Find an Attorney"]
        TASK_CREATE_LIVING_WILL = ["Create a Living Will"]
        TASK_COMPLETE_HEALTHCARE_POA = ["Complete a Healthcare Power of Attorney"]
        TASK_ESTABLISH_POA = ["Establish Power of Attorney"]
        TASK_GATHER_IMPORTANT_DOCS = ["Gather Important Documents"]

        # ---------- (d) Organize/Share Documents (final mini-tree)
        d_yes_final = self._register_node(LegalDocumentsNode(
            question="Great, I’ll add this to your task list once the onboarding process is complete.",
            tasks=TASK_GATHER_IMPORTANT_DOCS,
            leaf_node="leaf"
        ), "d_yes_final")

        d_unsure_final = self._register_node(LegalDocumentsNode(
            question="That’s okay. We can revisit this at a future time.",
            tasks=[],
            leaf_node="leaf"
        ), "d_unsure_final")

        d_no_final = self._register_node(LegalDocumentsNode(
            question="No problem, if @name’s needs change, I’m always available for assistance.",
            tasks=[],
            leaf_node="leaf"
        ), "d_no_final")

        d_root = self._register_node(LegalDocumentsNode(
            question=("With legal plans, it’s important to have the various documents organized in a secure place and "
                      "shared with relevant parties including medical and financial institutions. \n\n Would you like support "
                      "here organizing your documents and managing next steps?"),
            options={
                "Yes": "d_yes_final",
                "I’m not sure": "d_unsure_final",
                "No": "d_no_final"
            }
        ), "d_root")

        # ---------- (c) Advanced Directive & POA (routes to d_root)
        c_yes_ack = self._register_node(LegalDocumentsNode(
            question="Okay, great! I’ve noted @name has completed this already.",
            path="d_root"   # auto-advance to (d)
        ), "c_yes_ack")

        c_no_ack = self._register_node(LegalDocumentsNode(
            question=("It will be important to work with @name and an attorney to complete these documents as "
                      "soon as possible. We will add this to your Task list with more instructions once the onboarding "
                      "process is complete."),
            tasks=TASK_COMPLETE_HEALTHCARE_POA + TASK_ESTABLISH_POA,
            path="d_root"
        ), "c_no_ack")

        c_unsure_ack = self._register_node(LegalDocumentsNode(
            question=("[That’s okay. I suggest consulting with @name about what legal planning has been completed, if any, "
                      "to determine what needs to be done.]/[That’s okay. We can come back to this at a future time.]"),
            path="d_root"
        ), "c_unsure_ack")

        c_root = self._register_node(LegalDocumentsNode(
            question=("Has @name completed an advanced directive and power of attorney or conservatorship?"),
            options={
                "Yes": "c_yes_ack",
                "No": "c_no_ack",
                "I’m not sure": "c_unsure_ack"
            }
        ), "c_root")

        # ---------- (b) Trust/Will (routes to c_root)
        b_yes_ack = self._register_node(LegalDocumentsNode(
            question="Great! I’ve noted @name has completed this already.",
            path="c_root"
        ), "b_yes_ack")

        b_no_ack = self._register_node(LegalDocumentsNode(
            question=("It will be important to work with @name and an attorney to complete these documents as soon as "
                      "possible. I’ll add this to your Task list with more instructions once the onboarding process is complete."),
            tasks=TASK_CREATE_LIVING_WILL,
            path="c_root"
        ), "b_no_ack")

        b_unsure_ack = self._register_node(LegalDocumentsNode(
            question=("That’s okay. I suggest consulting with @name about if a trust or will has been completed to determine "
                      "what needs to be done."),
            path="c_root"
        ), "b_unsure_ack")

        b_root = self._register_node(LegalDocumentsNode(
            question=("Has @name completed a trust or a will?"),
            options={
                "Yes": "b_yes_ack",
                "No": "b_no_ack",
                "I’m not sure": "b_unsure_ack"
            }
        ), "b_root")

        # ---------- (a) Attorney (routes to b_root)
        a_yes_ack = self._register_node(LegalDocumentsNode(
            question="Okay, great.",
            path="b_root"
        ), "a_yes_ack")

        a_no_ack = self._register_node(LegalDocumentsNode(
            question=("It will be important to work with @name and an attorney to develop an estate plan as soon as "
                      "possible. I can help you research and find one that fits your needs. I’ll add this to your Task list "
                      "with more instructions once we finish onboarding."),
            tasks=TASK_FIND_ATTORNEY,
            path="b_root"
        ), "a_no_ack")

        a_unsure_ack = self._register_node(LegalDocumentsNode(
            question=("That’s okay. I suggest consulting with @name about if a trust or will has been completed to determine "
                      "what needs to be done."),
            path="b_root"
        ), "a_unsure_ack")

        # ---------- Root (exact wording you requested; presents the (a) question directly)
        self.root = self._register_node(LegalDocumentsNode(
            question=("You’re doing great. Just a few more questions so I can tailor the right support for you.\n\n"
                      "A comprehensive estate plan is very important to have completed so that @name’s wishes can be honored. "
                      "This plan can include a will, a trust, power of attorney, and advanced healthcare directives. I’d like to know "
                      "if I can help you with these documents. And again, anything you share will only be used to support your care.\n\n"
                      "Does @name have an attorney to help with estate planning?"),
            options={
                "Yes": "a_yes_ack",
                "No": "a_no_ack",
                "I’m not sure": "a_unsure_ack"
            }
        ), "root")

    # ----------------------------
    # Next-question precomputation (same style as Medicare)
    # ----------------------------
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    # Follow any linear paths to the next actual prompt
                    while (option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)
                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)
                node.next_questions = next_questions


class MedicaidAssessmentNode:
    def __init__(
        self,
        question,
        options=None,
        tasks=None,
        condition=None,
        path=None,
        node_id=None,
        leaf_node=None,
        next_questions=None,
    ):
        self.question = question
        self.options = options or None      # Dict mapping answer text -> child node_id
        self.path = path
        self.tasks = tasks or None          # Tasks to add based on this node
        self.node_id = node_id              # Unique identifier
        self.leaf_node = leaf_node or None  # "leaf" marks terminal node
        self.next_questions = next_questions or None


class MedicaidAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person
        self._build_tree()
        self._build_next_questions()

    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    def _build_tree(self):
        # ----- Leaf nodes -----
        enrolled_yes_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "Okay, I am glad that @name is already enrolled in Medicaid. "
                    "There are a lot of potential benefits within Medicaid that @name might be eligible for. "
                    "I’ll add a task to your list to explore your Medicaid benefits."
                ),
                tasks=["Explore Medicaid Benefits"],
                leaf_node="leaf",
            ),
            "enrolled_yes_leaf",
        )

        apply_medicaid_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "I can help @name enroll in Medicaid and will send more information after we finish onboarding."
                ),
                tasks=["Apply to Medicaid"],
                leaf_node="leaf",
            ),
            "apply_medicaid_leaf",
        )

        income_eligible_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "Yes, @name is eligible for Medicaid. I will provide you with the information on how to apply "
                    "in your Tasks list once the onboarding process is complete."
                ),
                tasks=["Apply to Medicaid"],
                leaf_node="leaf",
            ),
            "income_eligible_leaf",
        )

        income_not_eligible_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "It looks like @name is not eligible. We can explore what other benefits @name might be eligible for."
                ),
                tasks=[],
                leaf_node="leaf",
            ),
            "income_not_eligible_leaf",
        )

        not_eligible_root_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "That’s okay. We can explore what other benefits @name might be eligible for."
                ),
                tasks=[],
                leaf_node="leaf",
            ),
            "not_eligible_root_leaf",
        )

        skip_question_leaf = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "That’s okay. You are not required to answer anything you do not feel comfortable with. "
                    "We can move forward to the next question."
                ),
                tasks=[],
                leaf_node="leaf",
            ),
            "skip_question_leaf",
        )

        # ----- Middle-level nodes -----
        # Follow-up asked when user says they (or @name) are eligible
        already_enrolled = self._register_node(
            MedicaidAssessmentNode(
                question="And are you already enrolled in Medicaid?",
                options={
                    "Yes": "enrolled_yes_leaf",
                    "No": "apply_medicaid_leaf",
                    "I don’t want to answer this question": "skip_question_leaf",
                },
            ),
            "already_enrolled",
        )

        # Follow-up when user says “I don’t know” about eligibility
        income_question = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "Do you know what @name’s annual income is? I can help you check if they’re eligible."
                ),
                # These options represent the outcome after checking income
                options={
                    "Eligible": "income_eligible_leaf",
                    "Not eligible": "income_not_eligible_leaf",
                    "I don’t want to answer this question": "skip_question_leaf",
                },
            ),
            "income_question",
        )

        # ----- Root node -----
        self.root = self._register_node(
            MedicaidAssessmentNode(
                question=(
                    "Is @name eligible for Medicaid? If @name meets the income requirements, they may qualify for "
                    "Medicaid and still keep Medicare benefits if they have them."
                ),
                options={
                    "Yes": "already_enrolled",
                    "No": "not_eligible_root_leaf",
                    "I'm not sure": "apply_medicaid_leaf",
                    "I don’t know": "income_question",
                    "I don’t want to answer this question": "skip_question_leaf",
                },
            ),
            "root",
        )

    def _build_next_questions(self):
        for node_id in list(self.node_registry.keys()):
            node = self.get_node(node_id)
            next_questions = []
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                self.node_registry[node_id] = node
                continue
            if node.options:
                for option in node.options.values():
                    option_node = self.get_node(option)
                    while (
                        option_node.options is None
                        and option_node.leaf_node != "leaf"
                        and option_node.path is not None
                    ):
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

        not_sure_eligible = self._register_node(MedicareAssessmentNode(
            question="[@name] may be eligible for Medicare if they have a qualifying diagnosis. \nI can help determine your Medicare eligibility and can provide more information after we complete onboarding.",
            tasks=["Determine Medicare Eligibility"],
            leaf_node="leaf"
        ), "not_sure_eligible")
        
        not_enrolled = self._register_node(MedicareAssessmentNode(
            question="Information on how to enroll in Medicare will be provided in @name's Task list once the onboarding process is complete",
            tasks=["Enroll in Medicare"],
            leaf_node="leaf"
        ), "not_enrolled")
        
        not_eligible = self._register_node(MedicareAssessmentNode(
            question="No worries, we can explore what other benefits or resources @name might be eligible for.",
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
            question="Let’s talk about Medicare, the federal health insurance program. \nIs @name eligible for Medicare?",
            options={
                "Yes": "is_enrolled",
                "No": "not_eligible",
                "I'm not sure": "not_sure_eligible"
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
                question="Okay, I’ll add a task to help you evaluate veteran benefits and we can revisit it after we finish onboarding.",  # Terminal node with no further questions
                tasks=["Evaluate Veteran VA Benefits"],  # No additional tasks needed if already accessing benefits
                leaf_node="leaf"
            ),
            "accessing_benefits"
        )
        
        not_accessing_benefits = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="Okay, if anything changes I’m available 24/7 for assistance.",
                tasks=[],
                leaf_node="leaf"
            ),
            "not_accessing_benefits"
        )
        
        not_veteran = self._register_node(
            VeteranAssessmentNode(
                question="Would you like help exploring what benefits @name might be eligible for?",  # Terminal node with no further questions
                options={
                    "Yes": "yes_explore_benefits",
                    "No": "no_explore_benefits"
                },
                tasks=[],
                
            ),
            "not_veteran"
        )
        yes_explore_benefits = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="Okay, I’ll add a task to evaluate potential veteran benefits and we can revisit it after we finish onboarding.",
                tasks=["Evaluate Veteran VA Benefits"],
                leaf_node="leaf"
            ),
            "yes_explore_benefits"
        )
        
        no_explore_benefits = self._register_node(
            VeteranAssessmentNode(
                options=None,
                question="Okay, if anything changes I’m available 24/7 for assistance.",
                tasks=[],
                leaf_node="leaf"
            ),
            "no_explore_benefits"
        )        
        # Middle-level nodes
        is_accessing_benefits = self._register_node(
            VeteranAssessmentNode(
                question="Okay, great. Would you still like help exploring these benefits and other resources available for veterans?",
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
                question="You noted earlier that @name is a veteran and I’d like to see if there are additional resources or supportive services @name might be eligible for. \n\nIs @name currently accessing veterans benefits?",
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

class LiveSituationNode:
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None       # Dict[str, str] mapping choice -> child node_id
        self.tasks = tasks or None           # List[str] tasks to add at this node (leaf or mid)
        self.condition = condition           # Optional callable/flag for conditional routing
        self.path = path                     # Optional linear path pointer
        self.node_id = node_id               # Unique identifier
        self.leaf_node = leaf_node or None   # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Filled in after tree build


class LiveSituationAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # ----------------------------
    # Registry helpers
    # ----------------------------
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # ----------------------------
    # Tree construction
    # ----------------------------
    def _build_tree(self):
        # Common task labels
        TASK_MONITOR_CARE = ["Monitor In-Home or In-Facility Care"]
        TASK_FIND_INHOME_SUPPORT = ["Find In-Home Support"]

        # -------- Leaf nodes
        leaf_monitor_care = self._register_node(LiveSituationNode(
            question="Okay, I’ll add a task for this to your plan and we can discuss it more after we finish here.",
            tasks=TASK_MONITOR_CARE,
            leaf_node="leaf"
        ), "leaf_monitor_care")

        leaf_find_inhome_support = self._register_node(LiveSituationNode(
            question=("I’ll add a task to your plan to find in-home support and we can discuss it more after onboarding is complete. "
                      "WithCare also includes tools to help you request support from friends and family for tasks at home."),
            tasks=TASK_FIND_INHOME_SUPPORT,
            leaf_node="leaf"
        ), "leaf_find_inhome_support")

        leaf_no_questions_anytime = self._register_node(LiveSituationNode(
            question="No worries, if you have any questions later I’m available anytime.",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_no_questions_anytime")

        leaf_no_additional_support_needed = self._register_node(LiveSituationNode(
            question=("Okay I noted no additional support is needed right now. "
                      "If @name’s care needs change, I’m always available to provide assistance.\n"),
            tasks=[],
            leaf_node="leaf"
        ), "leaf_no_additional_support_needed")

        leaf_unsure_revisit = self._register_node(LiveSituationNode(
            question="No worries, we can always revisit this later.\n",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_unsure_revisit")

        leaf_unsure_general = self._register_node(LiveSituationNode(
            question="That’s okay, we can revisit this at a future time.\n",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_unsure_general")

        leaf_skip = self._register_node(LiveSituationNode(
            question=("No worries. You are not required to answer anything you do not feel comfortable with. "
                      "Let’s move on to the next question.\n"),
            tasks=[],
            leaf_node="leaf"
        ), "leaf_skip")

        # -------- ADL follow-ups (home branches)
        adl_assistance_home_alone = self._register_node(LiveSituationNode(
            question=("Does @name need assistance with activities of daily living such as dressing, "
                      "bathing or preparing meals?"),
            options={
                "Yes": "leaf_find_inhome_support",
                "No": "leaf_no_questions_anytime"
            }
        ), "adl_assistance_home_alone")

        adl_assistance_home_with_others = self._register_node(LiveSituationNode(
            question=("Does @name need assistance with activities of daily living such as dressing, "
                      "bathing or preparing meals?"),
            options={
                "Yes": "leaf_find_inhome_support",
                "No": "leaf_no_questions_anytime"
            }
        ), "adl_assistance_home_with_others")

        # -------- Home → Alone branch
        need_more_support_home = self._register_node(LiveSituationNode(
            question="Does @name need more support or care at home?",
            options={
                "Yes": "leaf_find_inhome_support",
                "I'm not sure": "adl_assistance_home_alone",
                "No": "leaf_no_questions_anytime"
            }
        ), "need_more_support_home")

        receives_inhome_care = self._register_node(LiveSituationNode(
            question="Does @name receive in-home care or assistance?",
            options={
                "Yes": "leaf_monitor_care",
                "No": "need_more_support_home"
            }
        ), "receives_inhome_care")

        # -------- Home → With others branch
        need_more_support_with_others = self._register_node(LiveSituationNode(
            question=("Great. Does @name need more support or care than is currently being provided "
                      "by those he/she live[s] with?"),
            options={
                "Yes": "leaf_find_inhome_support",
                "No": "leaf_no_additional_support_needed",
                "I'm not sure": "adl_assistance_home_with_others"
            }
        ), "need_more_support_with_others")

        # This node is used when user initially chooses “home” and says “With others”
        home_start_with_others = self._register_node(LiveSituationNode(
            question="And does @name live in this home alone or with others?",
            options={
                "With others": "need_more_support_with_others",
                "Alone": "receives_inhome_care"  # if they correct themselves
            }
        ), "home_start_with_others")

        # This node is used when user initially chooses “home” and says “Alone”
        home_start_alone = self._register_node(LiveSituationNode(
            question="And does @name live in this home alone or with others?",
            options={
                "Alone": "receives_inhome_care",
                "With others": "need_more_support_with_others"
            }
        ), "home_start_alone")

        # -------- Care facility branch
        facility_checkins = self._register_node(LiveSituationNode(
            question=("Would you like help checking in on how things are going with @name’s care? "
                      "I can schedule regular calls with the facility or caregivers to make sure everything’s on track "
                      "and keep you updated."),
            options={
                "Yes": "leaf_monitor_care",
                "No": "leaf_no_questions_anytime",
                "I'm not sure": "leaf_unsure_revisit"
            }
        ), "facility_checkins")

        # -------- Home vs Facility node (directly under root)
        home_or_facility = self._register_node(LiveSituationNode(
            question=("Examples of residences include a single family home, apartment, independent living, assisted living, "
                      "memory care, board and care or a skilled nursing facility.\n"
                      "Does @name live in a private residence home or in a care facility?"),
            options={
                "A private residence / single family home / apartment / independent living facility": "home_start_alone",
                "A care facility / assisted living / memory care / board and care / skilled nursing facility": "facility_checkins",
                "I'm not sure": "leaf_unsure_general",
                "I don’t want to answer this question": "leaf_skip"
            }
        ), "home_or_facility")

        # -------- Root node (exact wording you requested)
        self.root = self._register_node(LiveSituationNode(
            question=("I have a few questions about @name’s living situation so that I can provide recommendations "
                      "on things like care, safety, and planning for the future.\n"
                      "Examples of residences include a single family home, apartment, independent living, assisted living, "
                      "memory care, board and care or a skilled nursing facility.\n"
                      "Does @name live in a private residence home or in a care facility?"),
            options={
                # user taps an answer right on this question; it routes to the same choices as home_or_facility
                "A private residence / single family home / apartment / independent living facility": "home_start_alone",
                "A care facility / assisted living / memory care / board and care / skilled nursing facility": "facility_checkins",
                "I'm not sure": "leaf_unsure_general",
                "I don’t want to answer this question": "leaf_skip"
            }
        ), "root")

    # ----------------------------
    # Next-question precomputation (mirrors your Medicare approach)
    # ----------------------------
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    # Follow linear paths if used
                    while (option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)

                    if option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)

                node.next_questions = next_questions

class HospitalizationNode:
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None         # Dict[str, str]: choice -> child node_id
        self.tasks = tasks or None             # List[str]: tasks to add at this node
        self.condition = condition             # Optional: callable/flag for conditional routing
        self.path = path                       # Optional: linear path pointer (auto-advance)
        self.node_id = node_id                 # Unique identifier
        self.leaf_node = leaf_node or None     # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Populated after build


class HospitalizationAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # ----------------------------
    # Registry helpers
    # ----------------------------
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # ----------------------------
    # Tree construction
    # ----------------------------
    def _build_tree(self):
        # Task constants
        TASK_SUPPORT_DISCHARGE = ["Support with Discharge"]
        TASK_CHECKIN_REHAB = ["Check-in with Rehab Facility"]

        # ----- Generic leaves
        leaf_thanks = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me.",
            leaf_node="leaf",
            tasks=[]
        ), "leaf_thanks")

        leaf_skip = self._register_node(HospitalizationNode(
            question=("That’s okay. You are not required to answer anything you do not feel comfortable with. "
                      "We can move forward to the next question."),
            leaf_node="leaf",
            tasks=[]
        ), "leaf_skip")

        # ----- Root: No / I'm not sure → same note
        leaf_note_not_hosp = self._register_node(HospitalizationNode(
            question="Okay, I’ll note that they have not been hospitalized in the past 30 days.",
            leaf_node="leaf",
            tasks=[]
        ), "leaf_note_not_hosp")

        # =========================
        # YES FLOW
        # =========================

        # (1) Reason for hospitalization (free-text expected; then acknowledgment → destination)
        reason_ack = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me.",
            path="dest_home_rehab_still",   # auto-advance to destination
        ), "reason_ack")

        reason_prompt = self._register_node(HospitalizationNode(
            question="What was the reason for the hospitalization?",
            # No options: expects free-text; after capture your app should go to `reason_ack`
            path="reason_ack"
        ), "reason_prompt")

        # (2) Discharge destination
        dest_home_rehab_still = self._register_node(HospitalizationNode(
            question="Was @name discharged home or were they discharged to a rehabilitation facility?",
            options={
                "Home": "home_discharge_plan",
                "Rehab": "rehab_discharge_plan",
                "Still hospitalized": "still_discharge_plan"
            }
        ), "dest_home_rehab_still")

        # ----- HOME branch
        # (3H) Discharge plan provided?
        home_discharge_plan = self._register_node(HospitalizationNode(
            question="Was there a discharge plan provided?",
            options={
                "Yes": "home_confidence_support",
                "No": "home_need_additional_support"
            }
        ), "home_discharge_plan")

        # (4H-Yes) Confidence with instructions → regardless of answer, add Support with Discharge
        home_confidence_support = self._register_node(HospitalizationNode(
            question=("And are you confident in following the discharge instructions and managing their recovery at home?"),
            options={
                "Yes": "home_confidence_ack",
                "No": "home_confidence_ack",
                "I’m not sure": "home_confidence_ack"
            }
        ), "home_confidence_support")

        home_confidence_ack = self._register_node(HospitalizationNode(
            question=("I can support you with the discharge instructions and I will add this to your task list once we finish onboarding."),
            tasks=TASK_SUPPORT_DISCHARGE,
            leaf_node="leaf"
        ), "home_confidence_ack")

        # (4H-No) Additional support needed at home?
        home_need_additional_support = self._register_node(HospitalizationNode(
            question="Okay. Does @name need any additional support at home post-hospitalization?",
            options={
                "Yes": "home_support_yes_ack",
                "No": "home_support_no_ack"
            }
        ), "home_need_additional_support")

        home_support_yes_ack = self._register_node(HospitalizationNode(
            question=("Okay, thank you for telling me. I will follow up with you about this after we finish the onboarding questions."),
            leaf_node="leaf"
        ), "home_support_yes_ack")

        home_support_no_ack = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me.",
            leaf_node="leaf"
        ), "home_support_no_ack")

        # ----- REHAB branch
        # (3R) Discharge plan provided?
        rehab_discharge_plan = self._register_node(HospitalizationNode(
            question="Was there a discharge plan provided?",
            options={
                "Yes": "rehab_plan_yes_ack",
                "No": "rehab_need_additional_support"
            }
        ), "rehab_discharge_plan")

        rehab_plan_yes_ack = self._register_node(HospitalizationNode(
            question=("I can provide support and also help check-in with the facility staff about the discharge plan. "
                      "I will add this to your task list once we finish onboarding."),
            tasks=TASK_CHECKIN_REHAB + TASK_SUPPORT_DISCHARGE,
            leaf_node="leaf"
        ), "rehab_plan_yes_ack")

        rehab_need_additional_support = self._register_node(HospitalizationNode(
            question="Does @name need any additional support at home post-hospitalization?",
            options={
                "Yes": "rehab_support_yes_ack",
                "No": "rehab_support_no_ack",
            }
        ), "rehab_need_additional_support")

        rehab_support_yes_ack = self._register_node(HospitalizationNode(
            question=("Okay, thank you for telling me. I will follow up with you about this after we finish the onboarding questions."),
            leaf_node="leaf"
        ), "rehab_support_yes_ack")

        rehab_support_no_ack = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me.",
            leaf_node="leaf"
        ), "rehab_support_no_ack")

        # ----- STILL HOSPITALIZED branch
        # (3S) Discharge plan provided?
        still_discharge_plan = self._register_node(HospitalizationNode(
            question="Was there a discharge plan provided?",
            options={
                "Yes": "still_plan_yes_ack",
                "No": "still_need_additional_support"
            }
        ), "still_discharge_plan")

        still_plan_yes_ack = self._register_node(HospitalizationNode(
            question=("I can support you with the discharge instructions and I will add this to your task list once we finish onboarding."),
            tasks=TASK_SUPPORT_DISCHARGE,
            leaf_node="leaf"
        ), "still_plan_yes_ack")

        still_need_additional_support = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me. Does @name need any additional support at home post-hospitalization?",
            options={
                "Yes": "still_support_yes_ack",
                "No": "still_support_no_ack",
            }
        ), "still_need_additional_support")

        still_support_yes_ack = self._register_node(HospitalizationNode(
            question=("Okay, thank you for telling me. I will follow up with you about this after we finish the onboarding questions."),
            leaf_node="leaf"
        ), "still_support_yes_ack")

        still_support_no_ack = self._register_node(HospitalizationNode(
            question="Okay, thank you for telling me.",
            leaf_node="leaf"
        ), "still_support_no_ack")

        # =========================
        # ROOT (exact wording you gave)
        # =========================
        self.root = self._register_node(HospitalizationNode(
            question=("A hospitalization can cause a change in [@name’s]/[your] care needs. If this is relevant to you, "
                      "I’d like to know if I can provide assistance and help prevent future hospitalizations.\n\n"
                      "[Has @name]/[Have you] been hospitalized in the last 30 days? "
                      "Hospitalized means [@name/][you] stayed overnight and this does not count scheduled medical appointments or trips to urgent care"),
            options={
                "Yes": "reason_prompt",
                "No": "leaf_note_not_hosp",
                "I’m not sure": "leaf_note_not_hosp",
                "I don’t want to answer this question": "leaf_skip"
            }
        ), "root")

        # Link the intermediate node used above
        self._register_node(dest_home_rehab_still, "dest_home_rehab_still")
        self._register_node(home_discharge_plan, "home_discharge_plan")
        self._register_node(rehab_discharge_plan, "rehab_discharge_plan")
        self._register_node(still_discharge_plan, "still_discharge_plan")

    # ----------------------------
    # Next-question precomputation
    # ----------------------------
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    # Follow linear paths to the next actual prompt
                    while (option_node is not None and
                           option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)

                    if option_node is None or option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)

                node.next_questions = next_questions

class ERVisitNode:
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None         # Dict[str, str]: choice -> child node_id
        self.tasks = tasks or None             # List[str]
        self.condition = condition             # Optional callable/flag
        self.path = path                       # Optional linear path pointer (auto-advance)
        self.node_id = node_id                 # Unique identifier
        self.leaf_node = leaf_node or None     # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Filled after build


class ERVisitAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # Registry helpers
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # Tree construction
    def _build_tree(self):
        TASK_SUPPORT_WITH_DISCHARGE = ["Support with Discharge"]

        # ----- Generic leaves
        leaf_thanks = self._register_node(ERVisitNode(
            question="Okay, thank you for telling me.",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_thanks")

        leaf_revisit = self._register_node(ERVisitNode(
            question="That’s okay. We can revisit this at a future time.",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_revisit")

        leaf_skip = self._register_node(ERVisitNode(
            question=("That’s okay. You are not required to answer anything you do not feel comfortable with. "
                      "We can move forward to the next question."),
            tasks=[],
            leaf_node="leaf"
        ), "leaf_skip")

        # Root: No / I'm not sure → same note
        leaf_note_not_er = self._register_node(ERVisitNode(
            question="Okay, I’ll note that they have not visited the ER in the last 3 months.",
            tasks=[],
            leaf_node="leaf"
        ), "leaf_note_not_er")

        # =========================
        # YES FLOW
        # =========================

        # (1) Reason (free-text) → ack → admitted/discharged
        reason_ack = self._register_node(ERVisitNode(
            question="Okay, thank you for telling me.",
            path="admit_or_discharge"    # auto-advance
        ), "reason_ack")

        reason_prompt = self._register_node(ERVisitNode(
            question="What led to the ER visit?",
            path="reason_ack"            # collect free-text, then show ack, then auto-advance
        ), "reason_prompt")

        # (2) Admitted vs Discharged
        admit_or_discharge = self._register_node(ERVisitNode(
            question="Were @name admitted or discharged home?",
            options={
                "Admitted": "admitted_ack",
                "Discharged home": "discharge_plan_q"
            }
        ), "admit_or_discharge")

        admitted_ack = self._register_node(ERVisitNode(
            question="Okay, I’ve noted they were admitted.",
            leaf_node="leaf"
        ), "admitted_ack")

        # ----- Discharged home branch
        # (3) Discharge plan?
        discharge_plan_q = self._register_node(ERVisitNode(
            question="Was there a discharge plan provided?",
            options={
                "Yes": "support_following_discharge_q",
                "No": "no_discharge_plan_ack",
                "Not sure": "no_discharge_plan_ack"
            }
        ), "discharge_plan_q")

        no_discharge_plan_ack = self._register_node(ERVisitNode(
            question="Okay, I’ve noted for now that there was no discharge plan.",
            leaf_node="leaf"
        ), "no_discharge_plan_ack")

        # (4) Need support following discharge instructions?
        support_following_discharge_q = self._register_node(ERVisitNode(
            question="Does @name need any additional support following the discharge instructions?",
            options={
                "Yes": "support_yes_ack",
                "No": "support_no_ack"
            }
        ), "support_following_discharge_q")

        support_yes_ack = self._register_node(ERVisitNode(
            question=("Okay, thank you for telling me. I will follow up with you about this after we finish the onboarding questions."),
            tasks=TASK_SUPPORT_WITH_DISCHARGE,
            leaf_node="leaf"
        ), "support_yes_ack")

        support_no_ack = self._register_node(ERVisitNode(
            question="Okay, thank you for telling me.",
            leaf_node="leaf"
        ), "support_no_ack")

        # =========================
        # ROOT (exact wording you provided)
        # =========================
        self.root = self._register_node(ERVisitNode(
            question=("An ER visit may be a sign that @name’s health needs are changing. I’d like to know how I can "
                      "support you and help prevent future emergencies.\n\n"
                      "Has @name visited the ER in the last 3 months? An ER visit means they went to the "
                      "emergency room, even if they were sent home the same day."),
            options={
                "Yes": "reason_prompt",
                "No": "leaf_note_not_er",
                "I’m not sure": "leaf_note_not_er",
                "I don’t want to answer this question": "leaf_skip"
            }
        ), "root")

        # Register internal nodes referenced by path/ids above (defensive; already created)
        self._register_node(admit_or_discharge, "admit_or_discharge")
        self._register_node(discharge_plan_q, "discharge_plan_q")

    # Next-question precomputation (matches your pattern)
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    # Walk linear paths until a branching node or a leaf
                    while (option_node is not None and
                           option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)

                    if option_node is None or option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)

                node.next_questions = next_questions

class CopingNode:
    def __init__(self, question, options=None, tasks=None, condition=None,
                 path=None, node_id=None, leaf_node=None, next_questions=None):
        self.question = question
        self.options = options or None       # Dict[str, str]: choice -> child node_id
        self.tasks = tasks or None           # List[str]
        self.condition = condition           # Optional callable/flag
        self.path = path                     # Optional linear path pointer (auto-advance)
        self.node_id = node_id               # Unique identifier
        self.leaf_node = leaf_node or None   # "leaf" for terminal nodes
        self.next_questions = next_questions or None  # Filled after build


class CopingAssessmentTree:
    def __init__(self, person=None):
        self.node_registry = {}
        self.person = person or {}
        self._build_tree()
        self._build_next_questions()

    # ----------------------------
    # Registry helpers
    # ----------------------------
    def _register_node(self, node, node_id):
        node.node_id = node_id
        self.node_registry[node_id] = node
        return node

    def get_node(self, node_id):
        return self.node_registry.get(node_id)

    # ----------------------------
    # Tree construction
    # ----------------------------
    def _build_tree(self):
        # ---- Leaves / acknowledgments
        leaf_thanks_close = self._register_node(CopingNode(
            question=("Thank you, those are all of the questions I have. "
                      "I’ll send a message to you soon about your care plan tasks and next steps for @name’s care. You can exit now."),
            tasks=[],
            leaf_node="leaf"
        ), "leaf_thanks_close")

        # ---- First Coping Question (Yes path)
        # You can later extend this node with options or a path to subsequent questions in the series.
        coping_q1 = self._register_node(CopingNode(
            question="Do you have trouble concentrating throughout the day?",
            # No options specified yet in your spec; expects an answer your UI will collect.
            # You can wire a `path` here later to auto-advance to the next coping question.
            tasks=[],
            leaf_node="leaf"
        ), "coping_q1")

        # ---- Root (exact wording you requested)
        self.root = self._register_node(CopingNode(
            question=("I really appreciate you taking the time to answer all of these questions. Before we finish, I have a few "
                      "questions about how you have been coping with managing @name’s care. This will help me "
                      "to best support you in your caregiving journey.\n"
                      "Again, you are not required to answer anything you are not comfortable with, and anything you do answer is confidential. Are you okay to continue?"),
            options={
                "Yes": "coping_q1",
                "No": "leaf_thanks_close"
            }
        ), "root")

    # ----------------------------
    # Next-question precomputation (same style as your other trees)
    # ----------------------------
    def _build_next_questions(self):
        for node_id, node in self.node_registry.items():
            if node.leaf_node == "leaf":
                node.next_questions = ["None"]
                continue

            next_questions = []
            if node.options:
                for child_id in node.options.values():
                    option_node = self.get_node(child_id)
                    # Follow linear paths if you later add them
                    while (option_node is not None and
                           option_node.options is None and
                           option_node.leaf_node != "leaf" and
                           option_node.path is not None):
                        option_node = self.get_node(option_node.path)

                    if option_node is None or option_node.leaf_node == "leaf":
                        next_questions.append("")
                    else:
                        next_questions.append(option_node.question)

                node.next_questions = next_questions

class to_be_parsed_model(BaseModel):
    option: str = Field(None)
    has_additional_info: str = Field(None)

def assess_mental(state: GraphState, question_ls: list):
    client = genai.Client(api_key=get_gemini_api_key())
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
    print("[[[[[[[[[[]]]]]]]]]]")
    print(state.route)
    if state.route == "mental":
        return {"route_node": "assess_mental"}
    else:
        return {"route_node": "parse_response"}

def parse_response(state: GraphState, tree_dict: dict):
    # print(state['node'].options)
    # return {"options": state['node'].options, "node":list(state['node'].options.values())[0]}
    # print(state["node"], "////////")
    chat_history = state.chat_history
    real_chat_history = state.real_chat_history
    client = genai.Client(api_key=get_gemini_api_key())
    user_response = ""
    tasks = state.tasks
    for message in reversed(state.real_chat_history):
        if message["type"] == "human":
            user_response = message["content"] + user_response
        elif message["type"] == "ai":
            break

    if state.direct_record_answer:
        if state.care_time != True:
            print("direct_record_answer")
            identify_task_prompt = f"""
        Here is a conversation between an AI assitant and a caregiver user:
        {real_chat_history}

        Your task is to first detect whether the user has a task/request that needs to be helped with or not. And then, if the user has a task/request, you need to return the description of the task/request. If the user does not have a task/request, you need to return "None". IMPORTANT: you need to make sure that your output is in json format. There are two keys: "has_task", "task". The value of "has_task" should be "True" or "False". If the user has a task/request, then you must return "True" as the value for "has_task". If the user does not have a task/request, then you must return "False" as the value for "has_task". The value of "task" should be the description of the task/request. If the user does not have a task/request, then you must return "None" as the value for "task".

        Here is an example: 
        Example input (last two messages): 
        [{{"role": "assistant", "content": "Before we continue, is there anything specific about [@name's]/[your] care you need help figuring out or getting started on today?"}}, {{"role": "user", "content": "I want to know what are the good ways to take care of dementia people"}}]
        Example output: 
        {{"has_task": "True", "task": "yes,I want to know what are the good ways to take care of dementia people"}}

        Example input (last two messages): 
        [{{"role": "assistant", "content": "Before we continue, is there anything specific about [@name's]/[your] care you need help figuring out or getting started on today?"}}, {{"role": "user", "content": "Not really but i may have some later"}}]
        Example output: 
        {{"has_task": "False", "task": "None"}}
        """
            response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=identify_task_prompt,
            config={
                'response_mime_type': 'application/json',
    
            },
        )
            response_json = json.loads(response.text)
            has_task = response_json.get("has_task", "False")
            print("parsed response: ", response_json)
            task = response_json.get("task", "None")
            if has_task.lower() == "true":
                tasks.append(task)
                current_question = "Thanks for letting me know, I will mark this as top priority for the care plan. And I’ll start on it after onboarding." + "\n\n" + "Next, I am going to ask you a series of questions about @name’s care to better understand your situation and how I can best support you. \nYou are not required to answer anything you aren’t comfortable with, and anything you do answer is confidential and only used to help me provide the best assistance. \nHow long have you been providing care to @name?"
            else:
                current_question = "No problem! You can always reach out to me anytime you need help or have a question!" + "\n\n" + "Next, I am going to ask you a series of questions about @name’s care to better understand your situation and how I can best support you. \nYou are not required to answer anything you aren’t comfortable with, and anything you do answer is confidential and only used to help me provide the best assistance. \nHow long have you been providing care to @name?"
            return {"next_step": "ask_next_question", "last_step": "start", "question": current_question, "chat_history": chat_history + [HumanMessage(content=user_response),AIMessage(content=current_question)], "tasks": tasks, "care_time": True, "directly_ask": True, "node": "root", "current_tree": "MedicareAssessmentTree"}
        else:
            print("get_care_time")
            current_question = "Okay, thank you for sharing that.\n\nLet’s talk about Medicare, the federal health insurance program. \n\n[Is @name]/[Are you] eligible for Medicare?\n"
            return {"next_step": "ask_next_question", "last_step": "start", "question": current_question, "chat_history": chat_history + [HumanMessage(content=user_response)], "care_time": False, "direct_record_answer": False, "current_tree":"MedicareAssessmentTree", "node": "root"}
        
    tree = tree_dict[state.current_tree]
    if state.current_tree == "IntroAssessmentTree":
        direct_record_answer = True
        directly_ask = True
    else:
        direct_record_answer = False
        directly_ask = False
    current_question = tree.get_node(state.node).question
    # print("Next Questions:", tree.get_node(state.node).next_questions)
    next_questions = tree.get_node(state.node).next_questions
    all_questions = "Current Question: " + current_question + "\nLater Questions: " + str(next_questions)

    chat_history = state.chat_history
    # print(tree.get_node(state.node).options)
    if tree.get_node(state.node).options:
        options = list(tree.get_node(state.node).options.keys())
    else:
        options = [tree.get_node(state.node).path]
    # print("options", options)

    print("all_questions", all_questions)
    print("user_response", user_response)

    prompt = f"""System: we have a series of questions to ask the user to help them onboard. Your input contains four things: user response, current question, options, and later questions. Your job is to understand the user's response to your current question, and return one of the options you are given that best represent the user's response. At the same time, you need to look at the later questions and return a "has_additional_info" flag indicating if there is additional information in the user's response that can answer later questions. The "has_additional_info" is either "True" or "False". If the user's response has information to answer one of the later questions, then you must return "True" as the value for "has_additional_info". If the user's response has no information to answer any of the later questions, then you must return "False" as the value for "has_additional_info". 
    Important: never return "answer not found" as the option when the user response is a simple yes or no, instead return the option that best represent the user's response.

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

    # print("response_dict", response_dict)
    if response_dict["option"] != "answer not found":
        parsed_option = response_dict["option"]
        has_additional_info = response_dict["has_additional_info"]
        chat_history += [AIMessage(content=current_question),HumanMessage(content=parsed_option)]
        # update the current node
        node_id = tree.get_node(state.node).options[parsed_option]
        current_node = tree.get_node(node_id)
        print("current_node", node_id)
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
                print("end of tree:", current_question)
                current_tree_name = list(tree_dict.keys())[idx]
                if state.veteranStatus != "Veteran" and current_tree_name == "VeteranAssessmentTree":
                    current_tree_name = list(tree_dict.keys())[idx+1]
                    return {"current_tree": current_tree_name, "next_step": "ask_next_question", "last_step": "start", "node": "root", "question": current_question,"chat_history": chat_history, "tasks": tasks, "direct_record_answer": direct_record_answer}
                
                return {"current_tree": current_tree_name, "next_step": "ask_next_question", "last_step": "start", "node": "root", "question": current_question,"chat_history": chat_history, "tasks": tasks, "direct_record_answer": direct_record_answer, "directly_ask": directly_ask}
            else:
                ### last question: copyingassessmenttree
                return {"next_step": "ask_next_question", "question": current_question, "tasks": tasks, "direct_record_answer": direct_record_answer, "directly_ask": directly_ask,"chat_history": state.chat_history + [AIMessage(content=current_question)], "route": "mental"}


        current_question = current_node.question
        options = current_node.options
        if has_additional_info.lower() == "true":
            return {"node": node_id, "chat_history": chat_history, "tasks": tasks, "options": options, "question":current_question, "next_step": "parse_response", "last_step": "parse_response", "direct_record_answer": direct_record_answer, "directly_ask": directly_ask}
        else:
            return {"node": node_id, "chat_history": chat_history, "tasks": tasks, "options": options, "question":current_question, "next_step": "ask_next_question", "last_step": "parse_response", "direct_record_answer": direct_record_answer, "directly_ask": directly_ask}
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
    return {"real_chat_history": new_history, "question": "I'm sorry, I didn't understand that. Please try again.", "next_step": None}
        
def completed_onboarding(state: GraphState):
    new_history = state.real_chat_history + [AIMessage(content="Have you been sleeping more or less often than usual? Yes, no, or sometimes?")]
    return {"real_chat_history": new_history, "question": "Have you been sleeping more or less often than usual? Yes, no, or sometimes?", "chat_history": state.chat_history + [AIMessage(content="Have you been sleeping more or less often than usual? Yes, no, or sometimes?")], "route": "mental", "mental_question": "Have you been sleeping more or less often than usual? Yes, no, or sometimes?"}

async def short_completed_node(state: GraphState):
    new_history = state.real_chat_history + [AIMessage(content="Thank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
    question = "Thank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!"
    chat_history = state.chat_history + [AIMessage(content="Thank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
    return {"real_chat_history": new_history, "question":question, "chat_history": chat_history, "completed_whole_process": True}

async def completed_whole(state: GraphState):
    assess_score = state.assessment_score
    if assess_score < 5:
        new_history = state.real_chat_history + [AIMessage(content="You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and let me know if you need help finding any resources. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
        question = "You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and let me know if you need help finding any resources. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!"
        chat_history = state.chat_history + [AIMessage(content="You appear to be managing your stress well but it is important to maintain good self-care and utilize your support network. Stay connected to your care team for support and let me know if you need help finding any resources. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
    elif assess_score >= 5 and assess_score < 11:
        new_history = state.real_chat_history + [AIMessage(content="Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
        question = "Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!"
        chat_history = state.chat_history + [AIMessage(content="Your stress levels appear to be moderate. Utilize your calendar and tasks list to stay organized while managing the various responsibilities of caregiving. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
    elif assess_score >= 11:
        new_history = state.real_chat_history + [AIMessage(content="Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask me for assistance with finding additional resources such as support groups or individual therapists. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]
        question = "Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask me for assistance with finding additional resources such as support groups or individual therapists. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!"
        chat_history = state.chat_history + [AIMessage(content="Your answers indicate that you may be experiencing a high level of caregiver burnout. Check in with your own medical provider or primary care doctor if you need more support, and you can ask me for assistance with finding additional resources such as support groups or individual therapists. \n\nThank you, those are all of the questions I have! I’ll send a message to you soon about your care plan tasks and next steps!")]

    return {"real_chat_history": new_history, "question":question, "chat_history": chat_history, "completed_whole_process": True}

def ask_next_question(state: GraphState, tree_dict: dict):
    # print(state["real_chat_history"], "\\\\\\\\\\")
    client = genai.Client(api_key=get_gemini_api_key())
    care_recipient = str(state.care_recipient)
    # current_question = "Is there any immediate need I can support you with right now in regards to @name's care?"
    # chat_history = [AIMessage(content="Is your dad eligible for Medicaid? Medicaid is a government program that provides healthcare coverage for low-income individuals.", role="assistant"), HumanMessage(content="Yes, he is", role="user")]
    # chat_history = []
    if state.route == "mental" and state.question=="Do you have trouble concentrating throughout the day? Yes, no, or sometimes?":
        return {"question": state.question,"mental_question": state.question, "real_chat_history": state.real_chat_history + [AIMessage(content=state.question)], "last_step":"start", "directly_ask": directly_ask}
        
    directly_ask = False
    if state.directly_ask:
        current_question = state.question
        directly_ask = False
    else:
        current_question = tree_dict[state.current_tree].get_node(state.node).question
        print("raw question: ", current_question)
        if state.route == "mental":
            current_question = state.question 
        elif state.node == "root":
            current_question = state.question + "\n" + current_question
    real_chat_history = state.real_chat_history
    ask_template = f"""System: your job is to ask the question to the care giver to help them onboard. Your input includes a question, the chat history, and the care recipient information. 

    You need to undertstand the relationship between the care giver you are talking to and the care recipient. Refer to the care recipient based on the relationship. If the relationship is "self" then refer to the care recipient as you. Also look at the background information in chat history, formulate the question accordingly.
    #####
    Example inputs 1:
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

    Example output 1:
    The output question you need to ask is: Does your brother need more support or care than is currently being provided by your family?

    Example input 2:
    The care recipient information:
    {{
  "address": "11650 National Boulevard, Los Angeles, California 90064, United States",
  "dateOfBirth": "1968-03-12",
  "dependentStatus": "Not a child/dependent",
  "firstName": "Bruno",
  "gender": "Male",
  "isSelf": false,
  "lastName": "Mars",
  "legalName": "Bruno Mars",
  "pronouns": "he/him/his",
  "relationship": "Dad",
  "veteranStatus": "Veteran"
}}
    Input question: It will be important to work with @name and an attorney to complete these documents as soon as possible. We will add this to your Task list with more instructions once the onboarding process is complete. Is @name eligible for Medicare? Medicare is a federal health insurance program for those aged 65 or older, or who have a qualifying disability or diagnosis of end-stage kidney disease or ALS. 

    Chat history: [AIMessage(content="Does your dad have a will, living will, or power of attorney in place?", role="assistant"), HumanMessage(content="No he doesn't", role="user")]

    Example output 2:
     The output question you need to ask is: It will be important to work with your dad and an attorney to complete these documents as soon as possible. We will add this to your Task list with more instructions once the onboarding process is complete. Is your dad eligible for Medicare? Medicare is a federal health insurance program for those aged 65 or older, or who have a qualifying disability or diagnosis of end-stage kidney disease or ALS. 

     Important Note: In the input question, if there are sentences before or after the question sentence, make sure to include them and polish them so that @name is replaced with proper language.

    #### Below is the real input. 
    The care recipient information:
    {care_recipient}
    The question is {current_question}.  
    Chat history: {real_chat_history}
    
    Generate your output in the json format below, do not include any other text. Important: if there are line breaks in the input, make sure to include it in the output.
    
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
    print("generated question: ", generated_question)
    real_chat_history.append(AIMessage(content=generated_question))
    

    return {"question": generated_question, "real_chat_history": real_chat_history, "last_step":"start", "directly_ask": directly_ask}

def create_graph():
    memory = MemorySaver()
    builder = StateGraph(GraphState)
    tree_dict = {
        "IntroAssessmentTree":IntroAssessmentTree(),
    "MedicareAssessmentTree":MedicareAssessmentTree(),
    "MedicaidAssessmentTree":MedicaidAssessmentTree(),
    "VeteranAssessmentTree":VeteranAssessmentTree(), 
    "LiveSituationAssessmentTree":LiveSituationAssessmentTree(),
    "LegalDocumentsTree":LegalDocumentsAssessmentTree(), 
    "HospitalizationAssessmentTree":HospitalizationAssessmentTree(),
    "ERVisitAssessmentTree":ERVisitAssessmentTree(),
    "EndOfLifeCareTree":EndOfLifeCareTree(), 
    "CopingAssessmentTree":CopingAssessmentTree()}
    mental_health_questions_ls = ["Do you have trouble concentrating throughout the day? Yes, no, or sometimes?","Have you been sleeping more or less often than usual? Yes, no, or sometimes?", "Do you feel lonely or isolated? Yes, no, or sometimes?", "Have you lost interest in activities that you used to enjoy? Yes, no, or sometimes?", "Do you feel anxious, or like you can’t stop worrying about things that might happen? Yes, no, or sometimes?", "Do you feel down, sad or depressed? Yes, no, or sometimes?"]
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
    builder.add_node("short_completed_node", short_completed_node)
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("parse_response", lambda x: x.next_step, {"ask_to_repeat": "ask_to_repeat", "ask_next_question": "ask_next_question", "parse_response": "parse_response", "completed_onboarding": "completed_onboarding", "short_completed": "short_completed_node"})
    builder.add_conditional_edges("router_node", lambda x: x.route_node, {"assess_mental": "assess_mental", "parse_response": "parse_response"})
    builder.add_conditional_edges("assess_mental", lambda x: x.next_step, {END: END, "completed_whole": "completed_whole"})
    builder.add_edge("completed_whole", END)
    builder.add_edge("completed_onboarding", END)

    builder.add_edge("ask_to_repeat", END)
    builder.add_edge("ask_next_question", END)
    graph = builder.compile()  # Checkpointer removed for LangGraph API

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

