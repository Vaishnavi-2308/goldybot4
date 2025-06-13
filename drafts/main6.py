"""
https://claude.ai/public/artifacts/414b12e3-5930-43f6-8678-f3e769202987

GoldyBot - University of Minnesota Twin Cities Assistant
======================================================

A multi-agent chatbot system specifically designed for the University of Minnesota
Twin Cities students, featuring persistent knowledge base and web search capabilities.

Named after Goldy Gopher, the beloved UMN mascot!

Dependencies:
pip install langgraph langchain-ollama langchain-core typing-extensions requests beautifulsoup4 sqlite3
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from typing_extensions import Literal
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dataclasses import dataclass, field
import threading
import time
from urllib.parse import quote_plus

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

# ==================== UNIVERSITY OF MINNESOTA KNOWLEDGE BASE ====================

UMN_KNOWLEDGE = {
    "admissions": {
        "undergraduate_requirements": {
            "gpa_minimum": 3.8,
            "sat_range": "1370-1480",
            "act_range": "30-33",
            "acceptance_rate": "75%",
            "required_courses": ["4 years English", "4 years Math including Pre-calculus", "3 years Science", "3 years Social Studies", "2 years World Language"],
            "application_deadline": "January 1 (priority), May 1 (final)",
            "decision_date": "March 31",
            "campus": "Twin Cities"
        },
        "graduate_requirements": {
            "gpa_minimum": 3.0,
            "gre_required": "Program dependent",
            "application_deadline": "Varies by program (typically December 1-February 1)",
            "decision_date": "Varies by program"
        },
        "international_requirements": {
            "toefl_minimum": 79,
            "ielts_minimum": 6.5,
            "duolingo_minimum": 105,
            "additional_docs": ["Financial statement", "Passport copy", "Academic transcripts", "WES evaluation for international transcripts"]
        }
    },
    "programs": {
        "computer_science": {
            "college": "College of Science and Engineering (CSE)",
            "degree_types": ["BS Computer Science", "BS Data Science", "MS", "PhD"],
            "specializations": ["Artificial Intelligence", "Human-Computer Interaction", "Software Engineering", "Bioinformatics", "Graphics & Games"],
            "credits_required": 120,
            "internship_encouraged": True,
            "career_outcomes": ["Software Engineer at Google/Microsoft", "Data Scientist", "Research Scientist", "Product Manager"],
            "notable_faculty": ["Prof. Maria Gini (AI)", "Prof. Shashi Shekhar (Spatial Computing)"],
            "research_areas": ["AI/ML", "Robotics", "Computer Vision", "Distributed Systems"]
        },
        "business": {
            "college": "Carlson School of Management",
            "degree_types": ["BSB", "MBA", "MS programs"],
            "specializations": ["Finance", "Marketing", "Supply Chain", "Entrepreneurship", "Consulting"],
            "credits_required": 120,
            "internship_required": "Strongly encouraged",
            "career_outcomes": ["Investment Banking", "Management Consulting", "Marketing Manager", "Supply Chain Analyst"],
            "rankings": "Top 25 public business school (US News)",
            "notable_programs": ["Medical Industry Leadership Institute", "Entrepreneurship programs"]
        },
        "engineering": {
            "college": "College of Science and Engineering",
            "departments": ["Aerospace", "Biomedical", "Chemical", "Civil", "Computer", "Electrical", "Mechanical"],
            "degree_types": ["BS", "MS", "PhD"],
            "accreditation": "ABET accredited",
            "research_funding": "$180+ million annually",
            "industry_partnerships": ["3M", "Medtronic", "General Mills", "Target"]
        },
        "liberal_arts": {
            "college": "College of Liberal Arts (CLA)",
            "departments": ["Psychology", "English", "History", "Political Science", "Economics", "Sociology"],
            "degree_types": ["BA", "BS", "MA", "MS", "PhD"],
            "study_abroad": "200+ programs in 60+ countries",
            "research_opportunities": "Undergraduate Research Opportunities Program (UROP)"
        }
    },
    "financial_aid": {
        "tuition_2024_25": {
            "resident_undergraduate": 15254,
            "nonresident_undergraduate": 35494,
            "resident_graduate": 19116,
            "nonresident_graduate": 28592
        },
        "room_board": 11500,
        "total_cost_estimate": {
            "resident": 28754,
            "nonresident": 48994
        },
        "scholarships": {
            "merit_based": {
                "presidential_scholarship": {"amount": "Full tuition", "criteria": "Top 1% of applicants"},
                "deans_scholarship": {"amount": 15000, "criteria": "Top 10% of applicants"},
                "maroon_and_gold_scholarship": {"amount": 10000, "criteria": "Academic excellence"},
                "national_merit": {"amount": 7500, "criteria": "National Merit Finalist"}
            },
            "need_based": {
                "promise_scholarship": {"description": "Covers full tuition for families earning <$50k"},
                "pell_grant": {"max_amount": 7395, "criteria": "Federal need-based"},
                "state_grant": {"max_amount": 12000, "criteria": "Minnesota residents"}
            }
        },
        "work_study": {
            "available": True,
            "average_award": 2500,
            "positions": "On-campus jobs in academic departments, libraries, recreation centers"
        }
    },
    "housing": {
        "residence_halls": {
            "superblock": {
                "halls": ["Centennial", "Frontier", "Territorial", "Pioneer"],
                "capacity": 2800,
                "amenities": ["Dining halls", "Study spaces", "Recreation areas", "Laundry"],
                "location": "East Bank campus",
                "popular_with": "Freshmen and sophomores"
            },
            "west_bank": {
                "halls": ["Middlebrook", "Wilkins"],
                "capacity": 850,
                "amenities": ["Cultural programming", "Honors housing", "Music practice rooms"],
                "location": "West Bank campus",
                "popular_with": "Honors students, arts students"
            },
            "st_paul": {
                "halls": ["Bailey", "Comstock"],
                "capacity": 600,
                "amenities": ["Quiet environment", "Close to CFANS", "Nature access"],
                "location": "St. Paul campus",
                "popular_with": "CFANS students, graduate students"
            },
            "apartments": {
                "commonwealth_terrace": {"capacity": 1400, "type": "Family and graduate housing"},
                "como_student_community": {"capacity": 300, "type": "Apartment-style upperclass housing"}
            }
        },
        "dining": {
            "meal_plans": {
                "unlimited": {"cost": 5100, "description": "Unlimited dining hall access plus $300 FlexDine"},
                "block_225": {"cost": 4800, "description": "225 meals per semester plus $300 FlexDine"},
                "block_150": {"cost": 4400, "description": "150 meals per semester plus $400 FlexDine"}
            },
            "dining_halls": ["Centennial Dining", "Comstock Dining", "Fresh Food Company", "Pioneer Dining"],
            "campus_restaurants": ["Starbucks", "Subway", "Panda Express", "Erbert & Gerbert's"]
        }
    },
    "campus_life": {
        "student_organizations": {
            "total_orgs": "800+",
            "categories": ["Academic", "Cultural", "Greek Life", "Recreation", "Service", "Special Interest"],
            "popular_orgs": [
                "Programming Activities Council (PAC)",
                "Student Government", 
                "Daily Minnesota (newspaper)",
                "KUOM Radio",
                "Dance Marathon"
            ]
        },
        "recreation": {
            "facilities": ["Recreation & Wellness Center", "Aquatic Center", "Fieldhouse", "Outdoor gear rental"],
            "intramurals": "50+ sports and activities",
            "club_sports": "40+ competitive club teams",
            "outdoor_program": "Rock climbing, kayaking, camping trips"
        },
        "arts_culture": {
            "venues": ["Northrop Auditorium", "Ted Mann Concert Hall", "Rarig Center"],
            "museums": ["Weisman Art Museum", "Bell Museum"],
            "student_arts": "200+ music ensembles, theater groups, dance companies"
        },
        "traditions": {
            "homecoming": "Largest student-run homecoming in the nation",
            "spring_jam": "Annual music festival",
            "welcome_week": "New student orientation events",
            "rivalry": "Paul Bunyan's Axe (vs Wisconsin), Floyd of Rosedale (vs Iowa)"
        }
    },
    "campus_services": {
        "academic_support": [
            "Student Writing Support",
            "SMART Learning Commons",
            "Supplemental Instruction",
            "Tutoring services",
            "Academic Success Coaching"
        ],
        "health_services": [
            "Boynton Health",
            "Mental Health & Counseling",
            "Disability Resource Center",
            "Student Counseling Services"
        ],
        "technology": [
            "MyU portal",
            "Canvas LMS",
            "UMN WiFi (eduroam)",
            "Computer labs in all colleges",
            "Technology Help Desk",
            "Free software (Microsoft Office, Adobe Creative Suite)"
        ],
        "transportation": [
            "Campus Connector (free campus shuttle)",
            "Metro Transit (free with U-Pass)",
            "Bike rental and repair",
            "Car sharing programs",
            "Multiple parking ramps"
        ],
        "libraries": {
            "main": "Wilson Library",
            "others": ["Walter Library (Science & Engineering)", "Bio-Medical Library", "Law Library"],
            "features": ["24/7 study spaces", "Group study rooms", "Research assistance", "Special collections"]
        }
    },
    "athletics": {
        "conference": "Big Ten",
        "mascot": "Goldy Gopher",
        "colors": "Maroon and Gold",
        "major_sports": ["Football (Huntington Bank Stadium)", "Basketball (Williams Arena)", "Hockey (3M Arena at Mariucci)", "Baseball (Siebert Field)"],
        "notable_achievements": ["21 NCAA team championships", "Hockey national championships", "Wrestling excellence"],
        "facilities": ["Athletes Village", "Sports & Health Center", "Golf Course"]
    },
    "location_info": {
        "campus_locations": {
            "east_bank": "Main campus with most colleges and departments",
            "west_bank": "Arts, humanities, and social sciences",
            "st_paul": "Agriculture, food, environmental, and natural resource sciences",
            "rochester": "Health sciences programs"
        },
        "twin_cities": {
            "minneapolis": "Urban campus integrated with city",
            "saint_paul": "Capitol city, government internships",
            "metro_area": "3.6 million people, major business center"
        },
        "weather": {
            "winter": "Cold with snow (Dec-Mar), avg low 10°F",
            "summer": "Warm and humid (Jun-Aug), avg high 83°F",
            "fall": "Beautiful autumn colors, perfect weather",
            "spring": "Variable, can be wet"
        }
    }
}

# ==================== PERSISTENT KNOWLEDGE BASE ====================

class PersistentKnowledgeBase:
    """Persistent SQLite-based knowledge base for University of Minnesota Twin Cities"""
    
    def __init__(self, db_path: str = "goldybot_knowledge.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        self._populate_initial_data()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    key_term TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL DEFAULT 1.0,
                    UNIQUE(category, subcategory, key_term)
                )
            ''')
            
            # Search history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    category TEXT,
                    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    results_found INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Knowledge gaps table (tracks what we need to search for)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    missing_info TEXT NOT NULL,
                    category TEXT,
                    priority INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def _populate_initial_data(self):
        """Populate database with University of Minnesota Twin Cities knowledge"""
        for category, data in UMN_KNOWLEDGE.items():
            self._store_recursive(category, None, data)
    
    def _store_recursive(self, category: str, subcategory: Optional[str], data: Any, source: str = "initial"):
        """Recursively store nested dictionary data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self._store_recursive(category, key, value, source)
                else:
                    self.store_knowledge(category, subcategory, key, str(value), source)
        elif isinstance(data, list):
            self.store_knowledge(category, subcategory, "items", json.dumps(data), source)
        else:
            self.store_knowledge(category, subcategory, "value", str(data), source)
    
    def store_knowledge(self, category: str, subcategory: Optional[str], key_term: str, content: str, source: str = "web_search", confidence: float = 1.0):
        """Store knowledge entry in the database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_entries 
                    (category, subcategory, key_term, content, source, last_updated, confidence_score)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (category, subcategory, key_term, content, source, confidence))
                conn.commit()
    
    def search_knowledge(self, category: str = None, subcategory: str = None, key_term: str = None, query: str = None) -> List[Dict[str, Any]]:
        """Search for knowledge entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if query:
                # Full-text search across all fields
                cursor.execute('''
                    SELECT category, subcategory, key_term, content, source, last_updated, confidence_score
                    FROM knowledge_entries 
                    WHERE content LIKE ? OR key_term LIKE ? OR category LIKE ?
                    ORDER BY confidence_score DESC, last_updated DESC
                ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            else:
                # Exact search
                conditions = []
                params = []
                
                if category:
                    conditions.append("category = ?")
                    params.append(category)
                if subcategory:
                    conditions.append("subcategory = ?")
                    params.append(subcategory)
                if key_term:
                    conditions.append("key_term = ?")
                    params.append(key_term)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                cursor.execute(f'''
                    SELECT category, subcategory, key_term, content, source, last_updated, confidence_score
                    FROM knowledge_entries 
                    WHERE {where_clause}
                    ORDER BY confidence_score DESC, last_updated DESC
                ''', params)
            
            results = cursor.fetchall()
            return [
                {
                    "category": row[0],
                    "subcategory": row[1],
                    "key_term": row[2],
                    "content": row[3],
                    "source": row[4],
                    "last_updated": row[5],
                    "confidence_score": row[6]
                }
                for row in results
            ]
    
    def log_knowledge_gap(self, missing_info: str, category: str = None, priority: int = 1):
        """Log when information is missing and needs to be searched"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO knowledge_gaps (missing_info, category, priority)
                    VALUES (?, ?, ?)
                ''', (missing_info, category, priority))
                conn.commit()
    
    def get_knowledge_gaps(self, resolved: bool = False, limit: int = 10) -> List[Dict[str, Any]]:
        """Get unresolved knowledge gaps for web searching"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, missing_info, category, priority, created_at
                FROM knowledge_gaps 
                WHERE resolved = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            ''', (resolved, limit))
            
            results = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "missing_info": row[1],
                    "category": row[2],
                    "priority": row[3],
                    "created_at": row[4]
                }
                for row in results
            ]
    
    def mark_gap_resolved(self, gap_id: int):
        """Mark a knowledge gap as resolved"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE knowledge_gaps SET resolved = TRUE WHERE id = ?
                ''', (gap_id,))
                conn.commit()

# ==================== WEB SEARCH INTEGRATION ====================

class WebSearchManager:
    """Manages web searches to fill knowledge gaps for University of Minnesota"""
    
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        self.kb = knowledge_base
        self.umn_search_terms = [
            "University of Minnesota Twin Cities",
            "UMN Twin Cities",
            "University of Minnesota Minneapolis",
            "Goldy Gopher"
        ]
    
    def search_and_store(self, query: str, category: str = None, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search the web for UMN-specific information and store results"""
        # Add UMN-specific context to search queries
        umn_query = f"{query} University of Minnesota Twin Cities"
        
        results = []
        
        try:
            # Try DuckDuckGo with UMN-specific query
            search_results = self._search_duckduckgo(umn_query, max_results)
            
            for result in search_results:
                # Verify this is actually about UMN
                if self._is_umn_related(result.get('content', '') + result.get('title', '')):
                    content = self._extract_content(result.get('url', ''))
                    if content:
                        # Store in knowledge base
                        self.kb.store_knowledge(
                            category=category or "web_search",
                            subcategory=query[:50],
                            key_term=result.get('title', 'umn_web_result'),
                            content=content[:2000],
                            source=result.get('url', 'web'),
                            confidence=0.8
                        )
                        
                        results.append({
                            "title": result.get('title', ''),
                            "url": result.get('url', ''),
                            "content": content[:500],
                            "stored": True,
                            "umn_verified": True
                        })
            
            # Log the search
            with sqlite3.connect(self.kb.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO search_history (query, category, results_found, success)
                    VALUES (?, ?, ?, ?)
                ''', (umn_query, category, len(results), len(results) > 0))
                conn.commit()
        
        except Exception as e:
            print(f"UMN Search error: {e}")
            with sqlite3.connect(self.kb.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO search_history (query, category, results_found, success)
                    VALUES (?, ?, 0, FALSE)
                ''', (umn_query, category))
                conn.commit()
        
        return results
    
    def _is_umn_related(self, text: str) -> bool:
        """Check if content is related to University of Minnesota"""
        text_lower = text.lower()
        umn_indicators = [
            "university of minnesota",
            "umn",
            "twin cities",
            "minneapolis campus",
            "goldy gopher",
            "maroon and gold",
            "big ten",
            "gopher",
            "boynton health",
            "carlson school",
            "college of science and engineering"
        ]
        
        return any(indicator in text_lower for indicator in umn_indicators)
    
    def _search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (simple approach)"""
        try:
            # Use DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Get abstract if available
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('AbstractText', 'DuckDuckGo Result'),
                        'url': data.get('AbstractURL', ''),
                        'content': data.get('Abstract', '')
                    })
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:max_results-1]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('Text', '')[:100],
                            'url': topic.get('FirstURL', ''),
                            'content': topic.get('Text', '')
                        })
                
                return results[:max_results]
        
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return []
    
    def _extract_content(self, url: str) -> str:
        """Extract text content from a webpage"""
        try:
            if not url:
                return ""
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:2000]  # Limit content length
        
        except Exception as e:
            print(f"Content extraction error: {e}")
        
        return ""
    
    def fill_knowledge_gaps(self, max_gaps: int = 5):
        """Automatically fill knowledge gaps using web search"""
        gaps = self.kb.get_knowledge_gaps(resolved=False, limit=max_gaps)
        
        for gap in gaps:
            print(f"Filling UMN knowledge gap: {gap['missing_info']}")
            results = self.search_and_store(
                gap['missing_info'], 
                gap['category']
            )
            
            if results:
                self.kb.mark_gap_resolved(gap['id'])
                print(f"Resolved gap with {len(results)} results")
            else:
                print("No results found for gap")
            
            # Small delay to be respectful to search services
            time.sleep(1)

# ==================== INTENT CLASSIFICATION ====================

class IntentClassifier:
    """Classifies user intents for routing"""
    
    def __init__(self):
        self.intent_keywords = {
            "admissions": ["apply", "application", "admission", "requirements", "deadline", "gpa", "sat", "act", "transcript", "visit", "tour"],
            "academics": ["program", "major", "course", "degree", "curriculum", "class", "schedule", "professor", "credits", "college", "cse", "cla", "carlson"],
            "financial": ["cost", "tuition", "scholarship", "financial aid", "fafsa", "payment", "money", "grant", "loan", "promise scholarship"],
            "housing": ["dorm", "housing", "residence", "room", "dining", "meal plan", "cafeteria", "roommate", "superblock", "middlebrook"],
            "campus_life": ["activities", "clubs", "organizations", "sports", "recreation", "events", "student life", "gopher", "athletics", "homecoming"],
            "technical": ["portal", "login", "password", "wifi", "computer", "software", "email", "technology", "myu", "canvas"],
            "career": ["job", "career", "internship", "employment", "resume", "interview", "networking", "alumni"]
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
    
    def __init__(self, name: str, system_prompt: str, knowledge_base: PersistentKnowledgeBase):
        self.name = name
        self.llm = ChatOllama(model="qwen3:1.7b", temperature=0.1)
        self.system_prompt = system_prompt
        self.kb = knowledge_base
        self.web_search = WebSearchManager(knowledge_base)
    
    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a response using the LLM with knowledge base integration"""
        # First, search our knowledge base
        kb_results = self.kb.search_knowledge(query=query)
        
        # If no relevant knowledge found, search the web
        if not kb_results or len(kb_results) < 2:
            print(f"Limited knowledge found for '{query}', searching web...")
            category = self._determine_category(query)
            web_results = self.web_search.search_and_store(query, category)
            
            if web_results:
                # Search again after web results are stored
                kb_results = self.kb.search_knowledge(query=query)
        
        # Add knowledge base results to context
        context['knowledge_base_results'] = kb_results[:5]  # Top 5 results
        context['total_kb_entries'] = len(kb_results)
        
        # Format the context safely (avoiding curly braces issues)
        context_str = self._format_context_safe(context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"User query: {query}\n\nAvailable Knowledge: {context_str}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        return response.content
    
    def _format_context_safe(self, context: Dict[str, Any]) -> str:
        """Format context safely to avoid template variable issues"""
        # safe_context = {}
        # for key, value in context.items():
        #     if isinstance(value, str):
        #         # Replace curly braces to avoid template issues
        #         safe_context[key] = value.replace('{', '(').replace('}', ')')
        #     elif isinstance(value, (list, dict)):
        #         # Convert to string representation
        #         safe_context[key] = str(value).replace('{', '(').replace('}', ')')
        #     else:
        #         safe_context[key] = str(value)
        
        # return json.dumps(safe_context, indent=2)

        safe_context = json.dumps(context, indent=2)
        safe_context = safe_context.replace("{", "{{").replace("}", "}}")

        return safe_context
    
    def _determine_category(self, query: str) -> str:
        """Determine the most likely category for a query"""
        query_lower = query.lower()
        
        category_keywords = {
            "admissions": ["apply", "application", "admission", "requirements", "deadline"],
            "academics": ["program", "major", "course", "degree", "curriculum"],
            "financial": ["cost", "tuition", "scholarship", "financial aid"],
            "housing": ["dorm", "housing", "residence", "dining"],
            "campus_life": ["activities", "clubs", "organizations", "sports"],
            "technical": ["portal", "login", "wifi", "computer", "software"],
            "career": ["job", "career", "internship", "employment"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return "general"

class AdmissionsAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Admissions Specialist. You help prospective Gophers with:
        - Application requirements and processes for UMN Twin Cities
        - Admission deadlines and timelines for all UMN colleges
        - Required documents, test scores, and GPA requirements
        - International student requirements for studying at UMN
        - Transfer credit policies and procedures
        - Campus tours and visit information
        
        Use the UMN knowledge base to provide accurate, specific information about the University of Minnesota Twin Cities.
        Be encouraging and supportive while being clear about requirements.
        Always mention relevant UMN traditions and the welcoming Gopher community.
        Reference specific UMN colleges (CSE, CLA, Carlson, CFANS, etc.) when appropriate."""
        
        super().__init__("UMN Admissions Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        # Add specific UMN admissions knowledge to context
        admissions_info = self.kb.search_knowledge(category="admissions")
        context["umn_admissions_data"] = admissions_info
        context["university"] = "University of Minnesota Twin Cities"
        context["mascot"] = "Goldy Gopher"
        return self.generate_response(query, context)

class AcademicAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Academic Programs Specialist. You help Gophers with:
        - Degree programs across all UMN Twin Cities colleges (CSE, CLA, Carlson, CFANS, etc.)
        - Course descriptions, prerequisites, and academic planning
        - Faculty research opportunities and graduate programs
        - Study abroad programs (200+ programs in 60+ countries)
        - Undergraduate Research Opportunities Program (UROP)
        - Academic success resources and tutoring
        
        Provide detailed information about UMN's world-class academic offerings.
        Highlight UMN's research university status and Big Ten academic excellence.
        Be enthusiastic about the opportunities available at the U of M.
        Reference specific UMN facilities like Wilson Library, Walter Library, and research centers."""
        
        super().__init__("UMN Academic Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        programs_info = self.kb.search_knowledge(category="programs")
        context["umn_programs_data"] = programs_info
        context["university"] = "University of Minnesota Twin Cities"
        context["research_classification"] = "R1 Research University"
        return self.generate_response(query, context)

class FinancialAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Financial Aid Specialist. You help Gophers with:
        - UMN-specific scholarships (Promise Scholarship, Presidential, Dean's, etc.)
        - Minnesota state financial aid programs and residency benefits
        - FAFSA applications and federal aid for UMN students
        - Cost breakdowns for UMN Twin Cities tuition and expenses
        - Work-study opportunities and campus employment
        - Payment plans and billing information
        
        Provide clear, accurate financial information specific to UMN Twin Cities.
        Emphasize UMN's commitment to affordability and the Promise Scholarship program.
        Be sensitive to financial concerns and highlight all available aid options.
        Reference current 2024-25 tuition rates and costs."""
        
        super().__init__("UMN Financial Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        financial_info = self.kb.search_knowledge(category="financial_aid")
        context["umn_financial_data"] = financial_info
        context["university"] = "University of Minnesota Twin Cities"
        context["state"] = "Minnesota"
        return self.generate_response(query, context)

class HousingAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Housing and Dining Specialist. You help Gophers with:
        - UMN residence halls (Superblock, Middlebrook, Bailey, etc.)
        - Campus locations (East Bank, West Bank, St. Paul campuses)
        - Dining halls and meal plans specific to UMN
        - Roommate matching and housing applications
        - Graduate and family housing options
        - Off-campus housing in Minneapolis/St. Paul area
        
        Provide detailed information about living as a Gopher on campus.
        Help students understand the different campus areas and their unique features.
        Be welcoming and help them envision their life in the Twin Cities.
        Reference specific UMN dining locations and residence hall communities."""
        
        super().__init__("UMN Housing Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        housing_info = self.kb.search_knowledge(category="housing")
        context["umn_housing_data"] = housing_info
        context["university"] = "University of Minnesota Twin Cities"
        context["campus_locations"] = "East Bank, West Bank, St. Paul"
        return self.generate_response(query, context)

class CampusLifeAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Campus Life Specialist. You help Gophers with:
        - Student organizations (800+ registered student groups)
        - UMN traditions (Homecoming, Spring Jam, rivalry games)
        - Recreation and wellness (Rec Center, intramurals, club sports)
        - Arts and culture (Northrop, Weisman Museum, student performances)
        - Gopher athletics and Big Ten sports
        - Twin Cities cultural opportunities and internships
        
        Help students get excited about the vibrant Gopher community.
        Highlight UMN's unique traditions and the benefits of being in the Twin Cities.
        Be enthusiastic about Gopher pride and the maroon and gold spirit.
        Reference specific UMN venues, events, and Minneapolis/St. Paul opportunities."""
        
        super().__init__("UMN Campus Life Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        services_info = self.kb.search_knowledge(category="campus_life")
        athletics_info = self.kb.search_knowledge(category="athletics")
        context["umn_campus_life_data"] = services_info
        context["umn_athletics_data"] = athletics_info
        context["university"] = "University of Minnesota Twin Cities"
        context["mascot"] = "Goldy Gopher"
        context["colors"] = "Maroon and Gold"
        return self.generate_response(query, context)

class TechnicalAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Technical Support Specialist. You help Gophers with:
        - MyU portal access and student account setup
        - UMN email and Internet ID (x500) creation
        - Campus WiFi (eduroam) and network connectivity
        - Canvas learning management system navigation
        - UMN software licensing and computer lab access
        - Technology Help Desk and IT support resources
        
        Provide clear, step-by-step technical guidance for UMN systems.
        Be patient and thorough in explanations of UMN-specific technology.
        Reference UMN Technology Help and OIT (Office of Information Technology).
        Know when to escalate complex technical issues to UMN IT support."""
        
        super().__init__("UMN Technical Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        tech_info = self.kb.search_knowledge(category="campus_services", subcategory="technology")
        context["umn_tech_data"] = tech_info
        context["university"] = "University of Minnesota Twin Cities"
        context["help_desk"] = "UMN Technology Help"
        return self.generate_response(query, context)

class CareerAgent(BaseAgent):
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        system_prompt = """You are the University of Minnesota Twin Cities Career Services Specialist. You help Gophers with:
        - Career planning and exploration using UMN resources
        - Resume and cover letter assistance through Career Services
        - Interview preparation and networking opportunities
        - Internship searches in the Twin Cities and beyond
        - Job placement and alumni networking through UMN connections
        - Graduate school preparation and professional development
        
        Help students connect their UMN education to career opportunities.
        Highlight UMN's strong alumni network and Twin Cities business connections.
        Reference major Twin Cities employers (3M, Target, General Mills, Medtronic).
        Be encouraging about career prospects for UMN graduates."""
        
        super().__init__("UMN Career Agent", system_prompt, knowledge_base)
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        # Search for career-related information and Twin Cities job market
        career_info = self.kb.search_knowledge(query=f"career {query}")
        programs_info = self.kb.search_knowledge(category="programs")
        context["umn_career_data"] = career_info
        context["umn_programs_data"] = programs_info
        context["university"] = "University of Minnesota Twin Cities"
        context["location"] = "Twin Cities metro area"
        context["major_employers"] = ["3M", "Target", "General Mills", "Medtronic", "Best Buy"]
        return self.generate_response(query, context)

# ==================== ORCHESTRATOR ====================

class CentralOrchestrator:
    """Central orchestrator for managing GoldyBot conversation flow"""
    
    def __init__(self, knowledge_base: PersistentKnowledgeBase):
        self.llm = ChatOllama(model="qwen3:1.7b", temperature=0.1)
        self.intent_classifier = IntentClassifier()
        self.kb = knowledge_base
        self.web_search = WebSearchManager(knowledge_base)
        self.agents = {
            "admissions": AdmissionsAgent(knowledge_base),
            "academics": AcademicAgent(knowledge_base),
            "financial": FinancialAgent(knowledge_base),
            "housing": HousingAgent(knowledge_base),
            "campus_life": CampusLifeAgent(knowledge_base),
            "technical": TechnicalAgent(knowledge_base),
            "career": CareerAgent(knowledge_base)
        }
    
    def classify_and_route(self, state: ConversationState) -> ConversationState:
        """Classify user intent and determine routing for UMN-specific queries"""
        last_message = state["messages"][-1].content
        intent = self.intent_classifier.classify_intent(last_message)
        
        # Check if we have sufficient UMN knowledge for this query
        kb_results = self.kb.search_knowledge(query=last_message)
        if len(kb_results) < 2:
            # Log as knowledge gap for future UMN-specific web searches
            category = intent if intent != "general" else None
            umn_specific_query = f"{last_message} University of Minnesota Twin Cities"
            self.kb.log_knowledge_gap(umn_specific_query, category, priority=2)
        
        # Update state with intent and routing decision
        state["user_intent"] = intent
        state["current_agent"] = intent if intent != "general" else "admissions"
        
        return state
    
    def should_escalate(self, state: ConversationState) -> bool:
        """Determine if conversation should be escalated to human UMN staff"""
        last_message = state["messages"][-1].content.lower()
        
        escalation_triggers = [
            "speak to human", "talk to person", "not helpful",
            "frustrated", "angry", "complaint", "problem with",
            "admissions counselor", "academic advisor", "real person"
        ]
        
        return any(trigger in last_message for trigger in escalation_triggers)
    
    def run_background_knowledge_updates(self):
        """Run background process to fill UMN knowledge gaps"""
        print("Running background UMN knowledge updates...")
        self.web_search.fill_knowledge_gaps(max_gaps=3)

# ==================== LANGGRAPH NODES ====================

def orchestrator_node(state: ConversationState) -> ConversationState:
    """Central orchestrator node for routing and coordination"""
    # Initialize knowledge base if not exists in state
    if 'knowledge_base' not in state["conversation_context"]:
        kb = PersistentKnowledgeBase()
        state["conversation_context"]['knowledge_base'] = kb
        orchestrator = CentralOrchestrator(kb)
    else:
        kb = state["conversation_context"]['knowledge_base']
        orchestrator = CentralOrchestrator(kb)
    
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
    # Get knowledge base from state
    kb = state["conversation_context"].get('knowledge_base')
    if not kb:
        kb = PersistentKnowledgeBase()
        state["conversation_context"]['knowledge_base'] = kb
    
    orchestrator = CentralOrchestrator(kb)
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
        
        Was this helpful? Is there anything else I can help you with about the University of Minnesota Twin Cities? 🐿️
        I can assist with admissions, academics, financial aid, housing, campus life, or any other Gopher questions!
        
        💡 I'm continuously learning and updating my UMN knowledge base. If you need the most current information, 
        I can search for the latest Golden Gopher updates on any topic.
        
        *Go Gophers!* 🟤🟡
        """
        
        # Add satisfaction check to the last message
        last_message = state["messages"][-1]
        updated_content = last_message.content + satisfaction_prompt
        state["messages"][-1] = AIMessage(content=updated_content)
        
        state["satisfaction_collected"] = True
    
    state["next_action"] = "continue"
    return state

def escalation_node(state: ConversationState) -> ConversationState:
    """Handle escalation to human UMN support"""
    escalation_message = """
    I understand you'd like to speak with a human advisor from the University of Minnesota Twin Cities. 
    Let me connect you with one of our Gopher student services representatives who can provide personalized assistance.
    
    Here are the direct contact numbers for specific UMN Twin Cities departments:
    
    🏛️ **Admissions Office**: (612) 625-2008
       📧 admissions@umn.edu
    
    📚 **Academic Advising**: (612) 624-1111
       📧 onestop@umn.edu
    
    💰 **Financial Aid**: (612) 624-1665
       📧 finaid@umn.edu
    
    🏠 **Housing & Dining**: (612) 624-2994
       📧 housing@umn.edu
    
    💻 **Technology Help**: (612) 301-4357
       📧 help@umn.edu
    
    🎓 **Career Services**: (612) 624-3400
    
    You can also visit the OneStop Student Services center in 333 Robert H. Bruininks Hall 
    for in-person assistance with most student services.
    
    Would you like me to help you find more specific contact information or schedule an appointment?
    
    Go Gophers! 🐿️
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

def create_goldybot_chatbot() -> StateGraph:
    """Create the LangGraph workflow for GoldyBot - University of Minnesota Twin Cities assistant"""
    
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

# ==================== GOLDYBOT MAIN CLASS ====================

class GoldyBot:
    """GoldyBot - University of Minnesota Twin Cities Assistant with persistent knowledge base"""
    
    def __init__(self, db_path: str = "goldybot_knowledge.db"):
        self.knowledge_base = PersistentKnowledgeBase(db_path)
        self.web_search = WebSearchManager(self.knowledge_base)
        self.workflow = create_goldybot_chatbot()
        self.session_counter = 0
        
        # Run initial UMN knowledge gap filling
        print("🐿️ Initializing GoldyBot with University of Minnesota Twin Cities knowledge...")
        self.web_search.fill_knowledge_gaps(max_gaps=2)
    
    def start_conversation(self, user_id: str = None) -> tuple[str, str]:
        """Start a new conversation session with GoldyBot"""
        self.session_counter += 1
        session_id = f"goldybot_session_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get knowledge base stats
        all_knowledge = self.knowledge_base.search_knowledge()
        kb_stats = f"(Knowledge base: {len(all_knowledge)} UMN entries)"
        
        welcome_message = f"""
        🐿️ **Welcome to GoldyBot!** 🐿️
        *Your University of Minnesota Twin Cities Assistant*
        
        **SKI-U-MAH!** I'm here to help you navigate life as a Gopher! 🏛️✨
        
        I can assist you with everything about the **University of Minnesota Twin Cities**:
        
        📚 **Academic Programs** - CSE, CLA, Carlson, CFANS, and all UMN colleges
        📝 **Admissions** - Applications, requirements, and campus visits  
        💰 **Financial Aid** - Promise Scholarship, aid programs, and costs
        🏠 **Housing & Dining** - Superblock, residence halls, and meal plans
        🎯 **Campus Life** - 800+ student orgs, Gopher athletics, and Twin Cities fun
        💻 **Technology** - MyU portal, Canvas, WiFi, and UMN IT support
        🚀 **Career Services** - Internships, jobs, and alumni connections
        
        🔍 **Smart Gopher Knowledge**: I maintain current info about UMN Twin Cities and can search 
        for the latest Gopher news and updates when needed! {kb_stats}
        
        **Fun fact**: Did you know UMN has the largest student-run homecoming in the nation? 🎉
        
        What would you like to know about life as a Golden Gopher? Just ask me anything!
        
        *Go Gophers!* 🟤🟡
        """
        
        return welcome_message, session_id
    
    def chat(self, message: str, session_id: str, user_profile: Dict[str, Any] = None) -> str:
        """Process a chat message and return response from GoldyBot"""
        
        # Initialize state
        initial_state = ConversationState(
            messages=[HumanMessage(content=message)],
            current_agent=None,
            user_intent=None,
            user_profile=user_profile or {},
            conversation_context={"knowledge_base": self.knowledge_base},
            escalate_to_human=False,
            satisfaction_collected=False,
            session_id=session_id,
            next_action=None
        )
        
        try:
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract the response
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return """I'm sorry, I didn't quite understand that! 🐿️ 
                
                Could you please rephrase your question? I'm here to help with anything related to the University of Minnesota Twin Cities - from admissions and academics to campus life and Gopher athletics!
                
                *Go Gophers!* 🟤🟡"""
        
        except Exception as e:
            print(f"GoldyBot error: {e}")
            return """Oops! I'm experiencing some technical difficulties right now. 🐿️💻
            
            Please try again in a moment, or you can contact UMN Technology Help at (612) 301-4357 
            if you need immediate assistance.
            
            *Go Gophers!* 🟤🟡"""
    
    def update_umn_knowledge(self, category: str, subcategory: str, key_term: str, content: str, source: str = "manual"):
        """Manually update the UMN knowledge base"""
        self.knowledge_base.store_knowledge(category, subcategory, key_term, content, source)
    
    def search_umn_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search the UMN knowledge base"""
        return self.knowledge_base.search_knowledge(query=query)
    
    def get_goldybot_stats(self) -> Dict[str, Any]:
        """Get statistics about GoldyBot's knowledge base"""
        all_entries = self.knowledge_base.search_knowledge()
        gaps = self.knowledge_base.get_knowledge_gaps()
        
        return {
            "total_umn_entries": len(all_entries),
            "knowledge_categories": len(set(entry["category"] for entry in all_entries)),
            "unresolved_gaps": len(gaps),
            "web_sources": len([e for e in all_entries if e["source"] == "web_search"]),
            "university": "University of Minnesota Twin Cities",
            "mascot": "Goldy Gopher"
        }
    
    def run_umn_knowledge_update(self):
        """Manually trigger UMN knowledge base updates"""
        print("🐿️ Running University of Minnesota knowledge base updates...")
        self.web_search.fill_knowledge_gaps(max_gaps=5)
        stats = self.get_goldybot_stats()
        print(f"📊 GoldyBot knowledge updated: {stats}")

# ==================== EXAMPLE USAGE ====================

def main():
    """Example usage of GoldyBot - University of Minnesota Twin Cities assistant"""
    
    # Initialize GoldyBot
    print("🚀 Initializing GoldyBot for University of Minnesota Twin Cities...")
    goldybot = GoldyBot("goldybot_umn.db")
    
    # Display knowledge base stats
    stats = goldybot.get_goldybot_stats()
    print(f"📊 GoldyBot Knowledge Stats: {stats}")
    
    # Start conversation
    welcome_msg, session_id = goldybot.start_conversation()
    print("🐿️ GoldyBot:", welcome_msg)
    
    # Example conversation with UMN-specific queries
    test_queries = [
        "What are the admission requirements for computer science at UMN?",
        "How much does tuition cost for Minnesota residents?",
        "Tell me about housing in the Superblock",
        "What Gopher athletics can I attend as a student?",
        "How do I get involved in student organizations at UMN?",
        "What's the weather like in Minneapolis during winter?"
    ]
    
    user_profile = {
        "student_type": "prospective",
        "state_residency": "Minnesota",
        "interests": ["computer science", "Gopher hockey", "student organizations"],
        "communication_style": "friendly"
    }
    
    for i, query in enumerate(test_queries):
        print(f"\n👤 Prospective Gopher: {query}")
        response = goldybot.chat(query, session_id, user_profile)
        print(f"🐿️ GoldyBot: {response}")
        print("-" * 80)
        
        # Show knowledge base growth
        if i == 3:  # After a few queries
            new_stats = goldybot.get_goldybot_stats()
            print(f"📈 Updated GoldyBot Stats: {new_stats}")
    
    # Demonstrate manual UMN knowledge update
    print("\n🔧 Manually updating UMN knowledge base...")
    goldybot.update_umn_knowledge(
        category="academics",
        subcategory="computer_science", 
        key_term="new_ai_lab_2025",
        content="UMN CSE opened new AI Research Lab in Keller Hall featuring GPU clusters for machine learning research, available to graduate students and advanced undergraduates",
        source="umn_official_update"
    )
    
    # Test the updated knowledge
    print("\n👤 Prospective Gopher: Tell me about AI research opportunities at UMN")
    response = goldybot.chat("Tell me about AI research opportunities at UMN", session_id, user_profile)
    print(f"🐿️ GoldyBot: {response}")
    
    # Final knowledge base stats
    final_stats = goldybot.get_goldybot_stats()
    print(f"\n📊 Final GoldyBot Stats: {final_stats}")
    print("\n🐿️ SKI-U-MAH! Go Gophers! 🟤🟡")

if __name__ == "__main__":
    main()