"""
https://claude.ai/public/artifacts/907aa5ec-9c1b-4829-bc6f-092b95583e94

University Multi-Agent Chatbot System using LangGraph
=====================================================

A complete implementation of a multi-agent university assistant chatbot
using LangGraph for orchestration and state management.

Dependencies:
pip install langgraph langchain-openai langchain-core typing-extensions
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from typing_extensions import Literal
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import json
import re
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(os.path.dirname(__file__))

# ==================== STATE DEFINITIONS ====================

class ConversationState(TypedDict):
    """Main state for the conversation flow"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: Optional[str]
    user_intent: Optional[str]
    user_profile: Dict[str, Any]
    conversation_context: Dict[str, Any]
    escalate_to_human: bool
    satisfaction_collected: bool
    session_id: str
    next_action: Optional[str]

@dataclass
class UserProfile:
    """User profile for personalization"""
    student_id: Optional[str] = None
    student_type: Optional[str] = None  # prospective, new, continuing
    interests: List[str] = field(default_factory=list)
    communication_style: str = "balanced"  # brief, detailed, balanced
    language: str = "english"
    completed_topics: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

# ==================== UNIVERSITY KNOWLEDGE BASE ====================

UNIVERSITY_KNOWLEDGE = {
    "admissions": {
        "undergraduate_requirements": {
            "gpa_minimum": 3.0,
            "sat_minimum": 1200,
            "act_minimum": 26,
            "required_courses": ["4 years English", "3 years Math", "3 years Science", "3 years Social Studies"],
            "application_deadline": "January 15",
            "decision_date": "March 30"
        },
        "graduate_requirements": {
            "gpa_minimum": 3.5,
            "gre_required": True,
            "application_deadline": "December 1",
            "decision_date": "February 15"
        },
        "international_requirements": {
            "toefl_minimum": 90,
            "ielts_minimum": 7.0,
            "additional_docs": ["Financial statement", "Passport copy", "Academic transcripts"]
        }
    },
    "programs": {
        "computer_science": {
            "degree_types": ["BS", "MS", "PhD"],
            "specializations": ["AI/ML", "Cybersecurity", "Software Engineering", "Data Science"],
            "credits_required": 120,
            "internship_required": True,
            "career_outcomes": ["Software Engineer", "Data Scientist", "Product Manager"]
        },
        "business": {
            "degree_types": ["BBA", "MBA"],
            "specializations": ["Finance", "Marketing", "Management", "Entrepreneurship"],
            "credits_required": 120,
            "internship_required": False,
            "career_outcomes": ["Business Analyst", "Marketing Manager", "Financial Advisor"]
        }
    },
    "financial_aid": {
        "scholarships": {
            "merit_based": {
                "presidential_scholarship": {"amount": 20000, "gpa_required": 3.8},
                "dean_scholarship": {"amount": 15000, "gpa_required": 3.5},
                "honor_scholarship": {"amount": 10000, "gpa_required": 3.2}
            },
            "need_based": {
                "pell_grant": {"max_amount": 7000, "fafsa_required": True},
                "state_grant": {"max_amount": 5000, "residency_required": True}
            }
        },
        "costs": {
            "tuition_in_state": 12000,
            "tuition_out_state": 28000,
            "room_board": 14000,
            "books_supplies": 1200,
            "personal_expenses": 2000
        }
    },
    "housing": {
        "residence_halls": {
            "freshman_hall": {"capacity": 400, "amenities": ["Dining hall", "Study rooms", "Laundry"]},
            "sophomore_hall": {"capacity": 300, "amenities": ["Kitchen", "Lounge", "Gym"]},
            "upperclass_apartments": {"capacity": 200, "amenities": ["Full kitchen", "Living room", "Parking"]}
        },
        "meal_plans": {
            "unlimited": {"cost": 4500, "description": "Unlimited dining hall access"},
            "15_meals": {"cost": 4000, "description": "15 meals per week"},
            "10_meals": {"cost": 3500, "description": "10 meals per week plus dining dollars"}
        }
    },
    "campus_services": {
        "academic_support": ["Tutoring Center", "Writing Center", "Math Lab", "Science Learning Center"],
        "health_services": ["Student Health Center", "Counseling Services", "Disability Services"],
        "recreation": ["Fitness Center", "Swimming Pool", "Intramural Sports", "Outdoor Recreation"],
        "technology": ["Computer Labs", "WiFi Campus-wide", "Software Licensing", "Help Desk"]
    }
}

# ==================== INTENT CLASSIFICATION ====================

class IntentClassifier:
    """Classifies user intents for routing"""
    
    def __init__(self):
        self.intent_keywords = {
            "admissions": ["apply", "application", "admission", "requirements", "deadline", "gpa", "sat", "act", "transcript"],
            "academics": ["program", "major", "course", "degree", "curriculum", "class", "schedule", "professor", "credits"],
            "financial": ["cost", "tuition", "scholarship", "financial aid", "fafsa", "payment", "money", "grant", "loan"],
            "housing": ["dorm", "housing", "residence", "room", "dining", "meal plan", "cafeteria", "roommate"],
            "campus_life": ["activities", "clubs", "organizations", "sports", "recreation", "events", "student life"],
            "technical": ["portal", "login", "password", "wifi", "computer", "software", "email", "technology"],
            "career": ["job", "career", "internship", "employment", "resume", "interview", "networking"]
        }
    
    def classify_intent(self, message: str) -> str:
        """Classify the user's intent based on keywords"""
        message_lower = message.lower()
        
        # Score each intent category
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                scores[intent] = score
        
        # Return highest scoring intent or 'general' if no matches
        if scores:
            return max(scores, key=scores.get)
        return "general"

# ==================== SPECIALIST AGENTS ====================

class BaseAgent:
    """Base class for all specialist agents"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        # self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.llm = ChatOllama(model="qwen3:1.7b", temperature=0.1)
        self.system_prompt = system_prompt
    
    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a response using the LLM"""
        context_str = json.dumps(context, indent=2)
        context_str = context_str.replace("{", "{{").replace("}", "}}")
        human_str = f"User query: {query}\nContext: {context_str}"
        prompt = ChatPromptTemplate([
            ("system", self.system_prompt),
            ("human", human_str)
        ])
        
        chain = prompt | self.llm
        # response = chain.invoke({"query": query, "context": context})
        # response = chain.invoke({'\n  "user_profile"': context["user_profile"]})
        response = chain.invoke({})
        return response.content

class AdmissionsAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the University Admissions Specialist. You help prospective students with:
        - Application requirements and processes
        - Admission deadlines and timelines  
        - Required documents and test scores
        - International student requirements
        - Transfer credit policies
        
        Use the university knowledge base to provide accurate, specific information.
        Be encouraging and supportive while being clear about requirements.
        Always provide next steps and deadlines when relevant."""
        
        super().__init__("Admissions Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        admissions_info = UNIVERSITY_KNOWLEDGE["admissions"]
        context["admissions_data"] = admissions_info
        return self.generate_response(query, context)

class AcademicAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Academic Programs Specialist. You help students with:
        - Degree program information and requirements
        - Course descriptions and prerequisites
        - Academic planning and scheduling
        - Faculty and research opportunities
        - Study abroad and special programs
        
        Provide detailed information about academic offerings.
        Help students understand program requirements and career outcomes.
        Be enthusiastic about the academic opportunities available."""
        
        super().__init__("Academic Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        programs_info = UNIVERSITY_KNOWLEDGE["programs"]
        context["programs_data"] = programs_info
        return self.generate_response(query, context)

class FinancialAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Financial Aid Specialist. You help students with:
        - Scholarship opportunities and requirements
        - Financial aid applications (FAFSA, etc.)
        - Cost breakdowns and payment options
        - Work-study and employment opportunities
        - Budget planning and financial literacy
        
        Provide clear, accurate financial information.
        Help students understand all available financial aid options.
        Be sensitive to financial concerns and provide practical guidance."""
        
        super().__init__("Financial Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        financial_info = UNIVERSITY_KNOWLEDGE["financial_aid"]
        context["financial_data"] = financial_info
        return self.generate_response(query, context)

class HousingAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Housing and Dining Specialist. You help students with:
        - Residence hall options and amenities
        - Housing application processes
        - Meal plan selections
        - Roommate matching
        - Off-campus housing resources
        
        Provide detailed information about housing options.
        Help students make informed decisions about campus living.
        Be welcoming and help them envision their campus home."""
        
        super().__init__("Housing Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        housing_info = UNIVERSITY_KNOWLEDGE["housing"]
        context["housing_data"] = housing_info
        return self.generate_response(query, context)

class CampusLifeAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Campus Life Specialist. You help students with:
        - Student organizations and clubs
        - Campus events and activities
        - Recreation and fitness facilities
        - Student services and support
        - Campus traditions and culture
        
        Help students get excited about campus life opportunities.
        Provide information about getting involved and building community.
        Be enthusiastic and welcoming about the student experience."""
        
        super().__init__("Campus Life Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        services_info = UNIVERSITY_KNOWLEDGE["campus_services"]
        context["services_data"] = services_info
        return self.generate_response(query, context)

class TechnicalAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Technical Support Specialist. You help students with:
        - Student portal and system access
        - Email and account setup
        - WiFi and network connectivity
        - Software and technology resources
        - Basic troubleshooting
        
        Provide clear, step-by-step technical guidance.
        Be patient and thorough in explanations.
        Know when to escalate complex technical issues."""
        
        super().__init__("Technical Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        tech_info = UNIVERSITY_KNOWLEDGE["campus_services"]["technology"]
        context["tech_data"] = tech_info
        return self.generate_response(query, context)

class CareerAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are the Career Services Specialist. You help students with:
        - Career planning and exploration
        - Resume and cover letter assistance
        - Interview preparation
        - Internship and job search strategies
        - Networking and professional development
        
        Help students connect their academic interests to career opportunities.
        Provide practical advice for career preparation.
        Be encouraging about their future prospects."""
        
        super().__init__("Career Agent", system_prompt)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        # Extract career info from programs data
        career_info = {}
        for program, details in UNIVERSITY_KNOWLEDGE["programs"].items():
            career_info[program] = details.get("career_outcomes", [])
        context["career_data"] = career_info
        return self.generate_response(query, context)

# ==================== ORCHESTRATOR ====================

class CentralOrchestrator:
    """Central orchestrator for managing the conversation flow"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.llm = ChatOllama(model="qwen3:1.7b", temperature=0.1)
        self.intent_classifier = IntentClassifier()
        self.agents = {
            "admissions": AdmissionsAgent(),
            "academics": AcademicAgent(),
            "financial": FinancialAgent(),
            "housing": HousingAgent(),
            "campus_life": CampusLifeAgent(),
            "technical": TechnicalAgent(),
            "career": CareerAgent()
        }
    
    def classify_and_route(self, state: ConversationState) -> ConversationState:
        """Classify user intent and determine routing"""
        last_message = state["messages"][-1].content
        intent = self.intent_classifier.classify_intent(last_message)
        
        # Update state with intent and routing decision
        state["user_intent"] = intent
        state["current_agent"] = intent if intent != "general" else "admissions"
        
        return state
    
    def should_escalate(self, state: ConversationState) -> bool:
        """Determine if conversation should be escalated to human"""
        last_message = state["messages"][-1].content.lower()
        
        escalation_triggers = [
            "speak to human", "talk to person", "not helpful",
            "frustrated", "angry", "complaint", "problem with"
        ]
        
        return any(trigger in last_message for trigger in escalation_triggers)

# ==================== LANGGRAPH NODES ====================

def orchestrator_node(state: ConversationState) -> ConversationState:
    """Central orchestrator node for routing and coordination"""
    orchestrator = CentralOrchestrator()
    
    # Route the conversation
    state = orchestrator.classify_and_route(state)
    
    # Check for escalation
    if orchestrator.should_escalate(state):
        state["escalate_to_human"] = True
        state["next_action"] = "escalate"
    else:
        state["next_action"] = "specialist"
    
    return state

def specialist_agent_node(state: ConversationState) -> ConversationState:
    """Process query with the appropriate specialist agent"""
    orchestrator = CentralOrchestrator()
    current_agent = state["current_agent"]
    
    if current_agent not in orchestrator.agents:
        # Fallback to general assistance
        current_agent = "admissions"
    
    agent = orchestrator.agents[current_agent]
    last_message = state["messages"][-1].content
    
    # Prepare context for the agent
    context = {
        "user_profile": state["user_profile"],
        "conversation_history": [msg.content for msg in state["messages"][-5:]],
        "session_id": state["session_id"]
    }
    
    # Generate response
    response = agent.process_query(last_message, context)
    
    # Add agent response to messages
    state["messages"].append(AIMessage(content=response))
    
    # Update conversation context
    state["conversation_context"]["last_agent"] = current_agent
    state["conversation_context"]["topics_discussed"] = state["conversation_context"].get("topics_discussed", [])
    state["conversation_context"]["topics_discussed"].append(current_agent)
    
    state["next_action"] = "satisfaction_check"
    
    return state

def satisfaction_check_node(state: ConversationState) -> ConversationState:
    """Check if additional assistance is needed"""
    if not state["satisfaction_collected"]:
        satisfaction_prompt = """
        
        ---
        
        Was this helpful? Is there anything else I can help you with regarding your university experience? 
        I can assist with admissions, academics, financial aid, housing, campus life, or any other questions you might have.
        """
        
        # Add satisfaction check to the last message
        last_message = state["messages"][-1]
        updated_content = last_message.content + satisfaction_prompt
        state["messages"][-1] = AIMessage(content=updated_content)
        
        state["satisfaction_collected"] = True
    
    state["next_action"] = "continue"
    return state

def escalation_node(state: ConversationState) -> ConversationState:
    """Handle escalation to human support"""
    escalation_message = """
    I understand you'd like to speak with a human advisor. Let me connect you with one of our student services representatives who can provide personalized assistance.
    
    In the meantime, I can help you schedule an appointment or provide you with direct contact information for the specific department you need:
    
    - Admissions Office: (555) 123-4567
    - Academic Advising: (555) 123-4568  
    - Financial Aid: (555) 123-4569
    - Housing Services: (555) 123-4570
    
    Would you like me to help you schedule an appointment or provide more specific contact information?
    """
    
    state["messages"].append(AIMessage(content=escalation_message))
    state["next_action"] = "end"
    
    return state

def continuation_router(state: ConversationState) -> Literal["continue", "end"]:
    """Route based on whether conversation should continue"""
    if state["next_action"] == "end":
        return "end"
    
    # Check if user has follow-up questions
    if len(state["messages"]) > 0:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            return "continue"
    
    return "end"

def routing_decision(state: ConversationState) -> Literal["specialist", "escalate", "satisfaction_check"]:
    """Route based on orchestrator decision"""
    return state["next_action"]

# ==================== LANGGRAPH WORKFLOW ====================

def create_university_chatbot() -> StateGraph:
    """Create the LangGraph workflow for the university chatbot"""
    
    # Create the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("specialist", specialist_agent_node)
    workflow.add_node("satisfaction_check", satisfaction_check_node)
    workflow.add_node("escalation", escalation_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        routing_decision,
        {
            "specialist": "specialist",
            "escalate": "escalation",
            "satisfaction_check": "satisfaction_check"
        }
    )
    
    # Add edge from specialist to satisfaction check
    workflow.add_edge("specialist", "satisfaction_check")
    
    # Add conditional routing from satisfaction check
    workflow.add_conditional_edges(
        "satisfaction_check",
        continuation_router,
        {
            "continue": "orchestrator",
            "end": END
        }
    )
    
    # Add edge from escalation to end
    workflow.add_edge("escalation", END)
    
    return workflow.compile()

# ==================== MAIN CHATBOT CLASS ====================

class UniversityChatbot:
    """Main chatbot interface"""
    
    def __init__(self):
        self.workflow = create_university_chatbot()
        self.session_counter = 0
    
    def start_conversation(self, user_id: str = None) -> str:
        """Start a new conversation session"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        welcome_message = """
        🎓 Welcome to University Assistant! 🎓
        
        I'm here to help you with all aspects of your university experience. I can assist you with:
        
        📚 **Academic Programs** - Majors, courses, and degree requirements
        📝 **Admissions** - Application process, requirements, and deadlines  
        💰 **Financial Aid** - Scholarships, grants, and payment options
        🏠 **Housing & Dining** - Residence halls, meal plans, and campus living
        🎯 **Campus Life** - Clubs, activities, and student services
        💻 **Technical Support** - Student portal, email, and IT resources
        🚀 **Career Services** - Internships, job search, and career planning
        
        What would you like to know about? Just ask me anything!
        """
        
        return welcome_message, session_id
    
    def chat(self, message: str, session_id: str, user_profile: Dict[str, Any] = None) -> str:
        """Process a chat message and return response"""
        
        # Initialize state
        initial_state = ConversationState(
            messages=[HumanMessage(content=message)],
            current_agent=None,
            user_intent=None,
            user_profile=user_profile or {},
            conversation_context={},
            escalate_to_human=False,
            satisfaction_collected=False,
            session_id=session_id,
            next_action=None
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        # Extract the response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content
        else:
            return "I'm sorry, I didn't understand that. Could you please rephrase your question?"

# ==================== EXAMPLE USAGE ====================

def main():
    """Example usage of the university chatbot"""
    
    # Initialize chatbot
    chatbot = UniversityChatbot()
    
    # Start conversation
    welcome_msg, session_id = chatbot.start_conversation()
    print("🤖 Assistant:", welcome_msg)
    
    # Example conversation
    test_queries = [
        "What are the requirements to apply for computer science?",
        "How much does tuition cost?",
        "Tell me about housing options",
        "What clubs and activities are available?",
        "How do I apply for financial aid?"
    ]
    
    user_profile = {
        "student_type": "prospective",
        "interests": ["computer science", "technology"],
        "communication_style": "detailed"
    }
    
    for query in test_queries:
        print(f"\n👤 Student: {query}")
        response = chatbot.chat(query, session_id, user_profile)
        print(f"🤖 Assistant: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()