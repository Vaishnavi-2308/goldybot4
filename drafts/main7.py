"""
GoldyBot - Multi-Agent LangGraph Chatbot for University of Minnesota Students
"""

import asyncio
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
@dataclass
class BotConfig:
    model_name: str = "qwen3:1.7b"  # Change to your preferred Ollama model
    db_path: str = "goldybot.db"
    checkpoint_path: str = "goldybot_checkpoints.db"
    max_search_results: int = 5
    university_name: str = "University of Minnesota, Twin Cities"

config = BotConfig()

# State definition
class GoldyBotState(TypedDict):
    messages: Annotated[List, add_messages]
    user_id: str
    user_profile: Dict[str, Any]
    query_type: str
    search_needed: bool
    knowledge_found: bool
    context: str

# Database Manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge base table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    year TEXT,
                    major TEXT,
                    interests TEXT,
                    preferences TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Conversation history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message TEXT,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            conn.commit()
    
    def search_knowledge_base(self, query: str, category: str = None) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT question, answer, source, confidence 
                    FROM knowledge_base 
                    WHERE category = ? AND (question LIKE ? OR answer LIKE ?)
                    ORDER BY confidence DESC
                    LIMIT 5
                ''', (category, f'%{query}%', f'%{query}%'))
            else:
                cursor.execute('''
                    SELECT question, answer, source, confidence 
                    FROM knowledge_base 
                    WHERE question LIKE ? OR answer LIKE ?
                    ORDER BY confidence DESC
                    LIMIT 5
                ''', (f'%{query}%', f'%{query}%'))
            
            results = cursor.fetchall()
            return [
                {
                    'question': row[0],
                    'answer': row[1],
                    'source': row[2],
                    'confidence': row[3]
                }
                for row in results
            ]
    
    def add_knowledge(self, category: str, question: str, answer: str, source: str = None):
        """Add new knowledge to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO knowledge_base (category, question, answer, source)
                VALUES (?, ?, ?, ?)
            ''', (category, question, answer, source))
            conn.commit()
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, year, major, interests, preferences, interaction_count
                FROM user_profiles WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'name': result[0],
                    'year': result[1],
                    'major': result[2],
                    'interests': result[3],
                    'preferences': result[4],
                    'interaction_count': result[5]
                }
            return {}
    
    def update_user_profile(self, user_id: str, profile_data: Dict):
        """Update user profile in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT user_id FROM user_profiles WHERE user_id = ?', (user_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing profile
                cursor.execute('''
                    UPDATE user_profiles 
                    SET name=?, year=?, major=?, interests=?, preferences=?, 
                        interaction_count=interaction_count+1, updated_at=CURRENT_TIMESTAMP
                    WHERE user_id=?
                ''', (
                    profile_data.get('name'),
                    profile_data.get('year'),
                    profile_data.get('major'),
                    json.dumps(profile_data.get('interests', [])),
                    json.dumps(profile_data.get('preferences', {})),
                    user_id
                ))
            else:
                # Create new profile
                cursor.execute('''
                    INSERT INTO user_profiles 
                    (user_id, name, year, major, interests, preferences, interaction_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (
                    user_id,
                    profile_data.get('name'),
                    profile_data.get('year'),
                    profile_data.get('major'),
                    json.dumps(profile_data.get('interests', [])),
                    json.dumps(profile_data.get('preferences', {}))
                ))
            
            conn.commit()

# Initialize components
db_manager = DatabaseManager(config.db_path)
llm = ChatOllama(model=config.model_name, temperature=0.7)
search_tool = DuckDuckGoSearchRun()

# Tools
@tool
def search_university_info(query: str) -> str:
    """Search for University of Minnesota information online"""
    search_query = f"{query} University of Minnesota Twin Cities"
    try:
        results = search_tool.run(search_query)
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def query_knowledge_base(query: str, category: str = None) -> str:
    """Query the internal knowledge base"""
    results = db_manager.search_knowledge_base(query, category)
    if results:
        formatted_results = []
        for result in results:
            formatted_results.append(f"Q: {result['question']}\nA: {result['answer']}")
        return "\n\n".join(formatted_results)
    return "No relevant information found in knowledge base."

@tool
def update_user_info(user_id: str, info_type: str, value: str) -> str:
    """Update user profile information"""
    try:
        current_profile = db_manager.get_user_profile(user_id)
        current_profile[info_type] = value
        db_manager.update_user_profile(user_id, current_profile)
        return f"Updated {info_type} for user {user_id}"
    except Exception as e:
        return f"Failed to update user info: {str(e)}"

# Agent Nodes
async def classifier_agent(state: GoldyBotState) -> GoldyBotState:
    """Classify the user query and determine the appropriate response strategy"""
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier for GoldyBot, a University of Minnesota assistant.
        
        Classify the user query into one of these categories:
        - academics: Course information, registration, grades, requirements
        - campus_life: Dining, housing, events, activities, recreation
        - admissions: Application process, requirements, deadlines
        - financial: Tuition, financial aid, scholarships, billing
        - personal: User wants to share or update personal information
        - general: General university information, directions, contacts
        
        Also determine if online search is needed (true/false).
        
        Respond with: category|search_needed (e.g., "academics|true")"""),
        ("human", "{query}")
    ])
    
    chain = classify_prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"query": last_message})
    
    try:
        category, search_needed = result.strip().split("|")
        state["query_type"] = category
        state["search_needed"] = search_needed.lower() == "true"
    except:
        state["query_type"] = "general"
        state["search_needed"] = True
    
    return state

async def knowledge_agent(state: GoldyBotState) -> GoldyBotState:
    """Search the knowledge base for relevant information"""
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Search knowledge base
    kb_results = db_manager.search_knowledge_base(last_message, state["query_type"])
    
    if kb_results:
        state["knowledge_found"] = True
        state["context"] = f"Knowledge Base Results:\n{json.dumps(kb_results, indent=2)}"
    else:
        state["knowledge_found"] = False
        state["context"] = "No relevant information found in knowledge base."
    
    return state

async def search_agent(state: GoldyBotState) -> GoldyBotState:
    """Search online for information and update knowledge base"""
    
    if not state["search_needed"] and state["knowledge_found"]:
        return state
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Perform online search
    search_query = f"{last_message} University of Minnesota Twin Cities"
    try:
        search_results = search_tool.run(search_query)
        
        # Add to knowledge base
        db_manager.add_knowledge(
            category=state["query_type"],
            question=last_message,
            answer=search_results[:1000],  # Limit length
            source="DuckDuckGo Search"
        )
        
        # Update context
        if state["context"]:
            state["context"] += f"\n\nOnline Search Results:\n{search_results}"
        else:
            state["context"] = f"Online Search Results:\n{search_results}"
            
    except Exception as e:
        if not state["context"]:
            state["context"] = f"Search failed: {str(e)}"
    
    return state

async def personalization_agent(state: GoldyBotState) -> GoldyBotState:
    """Handle user personalization and profile management"""
    
    user_id = state["user_id"]
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Get current user profile
    user_profile = db_manager.get_user_profile(user_id)
    state["user_profile"] = user_profile
    
    # Extract personal information from the message
    personal_info_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract personal information from the user's message that could be stored in their profile.
        
        Look for:
        - Name
        - Year (freshman, sophomore, junior, senior, graduate)
        - Major/field of study
        - Interests
        - Preferences
        
        Respond with JSON format or "none" if no personal info found."""),
        ("human", "{message}")
    ])
    
    chain = personal_info_prompt | llm | StrOutputParser()
    extracted_info = await chain.ainvoke({"message": last_message})
    
    # Update profile if personal information is found
    if extracted_info.lower() != "none":
        try:
            info_dict = json.loads(extracted_info)
            current_profile = user_profile.copy()
            current_profile.update(info_dict)
            db_manager.update_user_profile(user_id, current_profile)
            state["user_profile"] = current_profile
        except:
            pass  # Failed to parse, continue without updating
    
    return state

async def response_agent(state: GoldyBotState) -> GoldyBotState:
    """Generate the final response to the user"""
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    user_profile = state.get("user_profile", {})
    context = state.get("context", "")
    
    # Create personalized response
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are GoldyBot, a helpful assistant for University of Minnesota, Twin Cities students.
        
        User Profile: {user_profile}
        Available Context: {context}
        
        Provide a helpful, personalized response. Be friendly and specific to UMN.
        If you don't have enough information, acknowledge this and suggest how to get more help.
        Always be encouraging and supportive of students' academic journey.
        
        Keep responses concise but informative. Use the user's name if available."""),
        ("human", "{query}")
    ])
    
    chain = response_prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "query": last_message,
        "user_profile": json.dumps(user_profile),
        "context": context
    })
    
    # Add AI response to messages
    state["messages"].append(AIMessage(content=response))
    
    return state

# Route condition
def should_search(state: GoldyBotState) -> str:
    """Determine if we need to search online"""
    if state["search_needed"] and not state["knowledge_found"]:
        return "search"
    return "personalization"

# Build the graph
def create_goldybot_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(GoldyBotState)
    
    # Add nodes
    workflow.add_node("classifier", classifier_agent)
    workflow.add_node("knowledge", knowledge_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("personalization", personalization_agent)
    workflow.add_node("response", response_agent)
    
    # Add edges
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "knowledge")
    workflow.add_conditional_edges(
        "knowledge",
        should_search,
        {
            "search": "search",
            "personalization": "personalization"
        }
    )
    workflow.add_edge("search", "personalization")
    workflow.add_edge("personalization", "response")
    workflow.add_edge("response", END)
    
    # Setup checkpointer
    # checkpointer = SqliteSaver.from_conn_string(config.checkpoint_path)
    
    # return workflow.compile(checkpointer=checkpointer)

    with SqliteSaver.from_conn_string(config.checkpoint_path) as checkpointer:
        return workflow.compile(checkpointer=checkpointer)


# Main GoldyBot class
class GoldyBot:
    def __init__(self):
        self.graph = create_goldybot_graph()
        self.db_manager = db_manager
    
    async def chat(self, message: str, user_id: str = "default_user") -> str:
        """Main chat interface"""
        
        # Create thread config
        thread_config = {"configurable": {"thread_id": f"thread_{user_id}"}}
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "user_profile": {},
            "query_type": "",
            "search_needed": False,
            "knowledge_found": False,
            "context": ""
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state, config=thread_config)
        
        # Return the last AI message
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        return ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't process your request."
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user interaction statistics"""
        profile = self.db_manager.get_user_profile(user_id)
        return {
            "profile": profile,
            "knowledge_base_size": self.get_knowledge_base_size(),
            "interaction_count": profile.get("interaction_count", 0)
        }
    
    def get_knowledge_base_size(self) -> int:
        """Get the size of the knowledge base"""
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
            return cursor.fetchone()[0]

# Example usage and testing
async def main():
    """Example usage of GoldyBot"""
    
    bot = GoldyBot()
    
    print("🐹 GoldyBot for University of Minnesota is ready!")
    print("Ask me anything about UMN - academics, campus life, admissions, and more!")
    print("Type 'quit' to exit\n")
    
    user_id = "student_123"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("GoldyBot: Goodbye! Go Gophers! 🐹")
                break
            
            if not user_input:
                continue
            
            print("GoldyBot: Thinking...")
            response = await bot.chat(user_input, user_id)
            print(f"GoldyBot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoldyBot: Goodbye! Go Gophers! 🐹")
            break
        # except Exception as e:
        #     print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())