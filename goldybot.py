#!/usr/bin/env python3
"""
GoldyBot - Multi-Agent LangGraph Chatbot for University of Minnesota Students
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Core dependencies
from langchain_core.messages import HumanMessage #, AIMessage, SystemMessage
from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph dependencies
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
# from langgraph.checkpoint.memory import MemorySaver

## Langsmith
from langsmith import traceable 
from langchain.callbacks.tracers.langchain import LangChainTracer

# Utility imports
import chromadb
from chromadb.config import Settings

load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(os.path.dirname(__file__))

class KnowledgeBase:
    """Simple persistent vector store using Chroma"""
    
    def __init__(self, db_path: str = "./knowledge_base"):
        self.db_path = db_path
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize or load the vectorstore
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the vectorstore, loading existing data if available"""
        try:
            # Try to get existing collection
            collection_name = "university_knowledge"
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing knowledge base from {self.db_path}")
        except Exception as e:
            # Create new collection if it doesn't exist
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="university_knowledge",
                embedding_function=self.embeddings
            )
            print(f"Created new knowledge base at {self.db_path}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base"""
        try:
            # Split documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            
            # Add to vectorstore
            if chunks:
                self.vectorstore.add_documents(chunks)
                print(f"Added {len(chunks)} chunks to knowledge base")
            
            # Persistence is automatic with ChromaDB PersistentClient
            
        except Exception as e:
            print(f"Error adding documents to knowledge base: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection = self.chroma_client.get_collection("university_knowledge")
            return {
                "count": collection.count(),
                "name": collection.name
            }
        except Exception as e:
            return {"count": 0, "name": "university_knowledge", "error": str(e)}


class UserProfile:
    """Manages user profile and preferences"""
    
    def __init__(self, user_id: str, profile_path: str = "./user_profiles"):
        self.user_id = user_id
        self.profile_path = profile_path
        self.profile_file = os.path.join(profile_path, f"{user_id}.json")
        
        # Ensure profile directory exists
        os.makedirs(profile_path, exist_ok=True)
        
        # Load or create profile
        self.profile = self._load_profile()
    
    def _load_profile(self) -> Dict[str, Any]:
        """Load user profile from file"""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading profile: {e}")
        
        # Default profile
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": {},
            "academic_info": {},
            "interests": [],
            "conversation_history": []
        }
    
    def save_profile(self):
        """Save profile to file"""
        try:
            self.profile["updated_at"] = datetime.now().isoformat()
            with open(self.profile_file, 'w') as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def update_field(self, field: str, value: Any):
        """Update a field in the user profile"""
        self.profile[field] = value
        self.save_profile()
    
    def add_to_history(self, interaction: Dict[str, Any]):
        """Add interaction to conversation history"""
        if "conversation_history" not in self.profile:
            self.profile["conversation_history"] = []
        
        self.profile["conversation_history"].append({
            **interaction,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 interactions
        if len(self.profile["conversation_history"]) > 50:
            self.profile["conversation_history"] = self.profile["conversation_history"][-50:]
        
        self.save_profile()


class GoldyBotState(TypedDict):
    """State for the GoldyBot conversation"""
    messages: Annotated[List[Any], "The conversation messages"]
    user_id: str
    query: str
    search_results: Optional[List[str]]
    knowledge_base_results: Optional[List[Document]]
    response: Optional[str]
    needs_search: bool
    user_info_extracted: Optional[Dict[str, Any]]


class GoldyBot:
    """Main GoldyBot class with multi-agent architecture"""
    
    def __init__(self, model_name: str = "qwen3:1.7b"):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        self.knowledge_base = KnowledgeBase()
        self.search_tool = DuckDuckGoSearchRun()
        
        # User profiles storage
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Initialize the graph
        self.graph = self._create_graph()
        # self.memory = MemorySaver()
        
        # Compile the graph
        # self.app = self.graph.compile(checkpointer=self.memory)
        self.app = self.graph.compile()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(GoldyBotState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("knowledge_search", self._search_knowledge_base)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("user_info_extractor", self._extract_user_info)
        workflow.add_node("response_generator", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("classifier")
        
        # Conditional edges from classifier
        workflow.add_conditional_edges(
            "classifier",
            self._route_after_classification,
            {
                "knowledge_search": "knowledge_search",
                "web_search": "web_search",
                "user_info": "user_info_extractor"
            }
        )
        
        # From knowledge_search
        workflow.add_conditional_edges(
            "knowledge_search",
            self._route_after_knowledge_search,
            {
                "web_search": "web_search",
                "response": "response_generator"
            }
        )
        
        # From web_search and user_info_extractor
        workflow.add_edge("web_search", "response_generator")
        workflow.add_edge("user_info_extractor", "response_generator")
        
        # End at response_generator
        workflow.add_edge("response_generator", END)
        
        return workflow
    
    def _classify_query(self, state: GoldyBotState) -> GoldyBotState:
        """Classify the user query to determine routing"""
        query = state["query"]
        
        classification_prompt = f"""
        You are a query classifier for a University of Minnesota student assistant bot.
        
        Classify this query into one of these categories:
        1. "knowledge_search" - Questions about university policies, academics, campus life, etc.
        2. "user_info" - User is providing personal information about themselves
        3. "web_search" - Questions requiring current information or not in knowledge base
        
        Query: {query}
        
        Respond with just the category name.
        """
        
        try:
            classification = self.llm.invoke(classification_prompt).content.strip().lower()
            
            if "user_info" in classification:
                route = "user_info"
            elif "web_search" in classification:
                route = "web_search"
                state["needs_search"] = True
            else:
                route = "knowledge_search"
                state["needs_search"] = False
                
            print(f"Query classified as: {route}")
            return state
            
        except Exception as e:
            print(f"Error in classification: {e}")
            state["needs_search"] = False
            return state
    
    def _search_knowledge_base(self, state: GoldyBotState) -> GoldyBotState:
        """Search the knowledge base for relevant information"""
        query = state["query"]
        
        try:
            results = self.knowledge_base.search(query, k=3)
            state["knowledge_base_results"] = results
            
            if results:
                print(f"Found {len(results)} relevant documents in knowledge base")
            else:
                print("No relevant documents found in knowledge base")
                state["needs_search"] = True
                
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            state["needs_search"] = True
            
        return state
    
    def _web_search(self, state: GoldyBotState) -> GoldyBotState:
        """Perform web search and update knowledge base"""
        query = state["query"]
        
        try:
            # Enhance search query for University of Minnesota
            enhanced_query = f"{query} University of Minnesota Twin Cities"
            
            search_results = self.search_tool.run(enhanced_query)
            state["search_results"] = [search_results]
            
            # Add to knowledge base
            if search_results:
                doc = Document(
                    page_content=search_results,
                    metadata={
                        "source": "web_search",
                        "query": query,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.knowledge_base.add_documents([doc])
                print("Added web search results to knowledge base")
            
        except Exception as e:
            print(f"Error in web search: {e}")
            state["search_results"] = []
            
        return state
    
    def _extract_user_info(self, state: GoldyBotState) -> GoldyBotState:
        """Extract and store user information"""
        query = state["query"]
        user_id = state["user_id"]
        
        extraction_prompt = f"""
        Extract any personal information from this user message that might be relevant for a university assistant bot.
        
        Look for:
        - Academic information (major, year, college)
        - Interests and preferences
        - Contact information
        - Goals and plans
        
        Message: {query}
        
        Return the information as a JSON object. If no relevant information is found, return an empty object.
        """
        
        try:
            extraction_result = self.llm.invoke(extraction_prompt).content
            extraction_result = extraction_result.split('</think>')[-1]
            
            # Try to parse as JSON
            try:
                user_info = json.loads(extraction_result)
                state["user_info_extracted"] = user_info
                
                # Update user profile
                if user_id in self.user_profiles:
                    profile = self.user_profiles[user_id]
                    for key, value in user_info.items():
                        if key in ["major", "year", "college"]:
                            profile.profile["academic_info"][key] = value
                        elif key == "interests":
                            profile.profile["interests"].extend(value if isinstance(value, list) else [value])
                        else:
                            profile.profile["preferences"][key] = value
                    profile.save_profile()
                    
            except json.JSONDecodeError:
                state["user_info_extracted"] = {}
                
        except Exception as e:
            print(f"Error extracting user info: {e}")
            state["user_info_extracted"] = {}
            
        return state
    
    def _generate_response(self, state: GoldyBotState) -> GoldyBotState:
        """Generate the final response"""
        query = state["query"]
        user_id = state["user_id"]
        
        # Get user profile
        user_profile = self.user_profiles.get(user_id)
        profile_context = ""
        if user_profile:
            profile_context = f"""
            User Profile Context:
            - Academic Info: {user_profile.profile.get('academic_info', {})}
            - Interests: {user_profile.profile.get('interests', [])}
            - Preferences: {user_profile.profile.get('preferences', {})}
            """
        
        # Prepare context from knowledge base and search results
        context = ""
        if state.get("knowledge_base_results"):
            context += "Knowledge Base Information:\n"
            for doc in state["knowledge_base_results"]:
                context += f"- {doc.page_content[:300]}...\n"
        
        if state.get("search_results"):
            context += "\nWeb Search Results:\n"
            for result in state["search_results"]:
                context += f"- {result[:300]}...\n"
        
        # Generate response
        response_prompt = f"""
        You are GoldyBot, a friendly and knowledgeable assistant for University of Minnesota, Twin Cities students.
        
        {profile_context}
        
        Context Information:
        {context}
        
        User Question: {query}
        
        Provide a helpful, personalized response. Be conversational and supportive. If you're using information from the context, make sure it's accurate and relevant. If you don't have enough information, be honest about it.
        
        Remember to:
        - Be specific to University of Minnesota when possible
        - Offer to help with follow-up questions
        - Be encouraging and supportive
        - Use a friendly, casual tone appropriate for college students
        """
        
        try:
            response = self.llm.invoke(response_prompt).content
            response = response.split('</think>')[-1]
            state["response"] = response
            
            # Add to user's conversation history
            if user_profile:
                user_profile.add_to_history({
                    "query": query,
                    "response": response,
                    "context_used": bool(context)
                })
                
        except Exception as e:
            print(f"Error generating response: {e}")
            state["response"] = "I'm sorry, I encountered an error while processing your request. Please try again."
            
        return state
    
    def _route_after_classification(self, state: GoldyBotState) -> str:
        """Route after query classification"""
        query = state["query"]
        
        # Simple keyword-based routing as fallback
        user_info_keywords = ["my", "i am", "i'm", "my major", "my year", "i study"]
        web_search_keywords = ["current", "latest", "recent", "today", "this year", "2024", "2025"]
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in user_info_keywords):
            return "user_info"
        elif any(keyword in query_lower for keyword in web_search_keywords):
            return "web_search"
        else:
            return "knowledge_search"
    
    def _route_after_knowledge_search(self, state: GoldyBotState) -> str:
        """Route after knowledge base search"""
        if state.get("needs_search", True) or not state.get("knowledge_base_results"):
            return "web_search"
        else:
            return "response"
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    
    @traceable
    async def chat(self, message: str, user_id: str = None) -> str:
        """Main chat interface"""
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        # Ensure user profile exists
        self.get_or_create_user_profile(user_id)
        
        # Create initial state
        initial_state = GoldyBotState(
            messages=[HumanMessage(content=message)],
            user_id=user_id,
            query=message,
            search_results=None,
            knowledge_base_results=None,
            response=None,
            needs_search=False,
            user_info_extracted=None
        )
        

        tracer = LangChainTracer()

        # Run the graph
        # config = {"configurable": {"thread_id": user_id}}
        config = {"configurable": {"thread_id": user_id}, "callbacks": [tracer]}
        
        try:
            final_state = await self.app.ainvoke(initial_state, config)
            return final_state.get("response", "I'm sorry, I couldn't process your request.")
        except Exception as e:
            print(f"Error in chat processing: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base"""
        return self.knowledge_base.get_collection_info()


# Example usage and testing
async def main():
    """Example usage of GoldyBot"""
    
    print("Initializing GoldyBot...")
    bot = GoldyBot()
    
    print(f"Knowledge base info: {bot.get_knowledge_base_info()}")
    
    # Test conversations
    test_queries = [
        "Hi, I'm a sophomore majoring in Computer Science. What programming courses should I take?",
        "What are the library hours at the University of Minnesota?",
        "I'm interested in undergraduate research opportunities. Can you help?",
        "What dining options are available on campus?",
        "How do I register for classes next semester?"
    ]
    
    user_id = "test_user_123"
    
    for query in test_queries:
        print(f"\n🎓 User: {query}")
        response = await bot.chat(query, user_id)
        print(f"🤖 GoldyBot: {response}")
        print("-" * 80)
    
    # Show updated knowledge base info
    print(f"\nFinal knowledge base info: {bot.get_knowledge_base_info()}")


if __name__ == "__main__":
    asyncio.run(main())