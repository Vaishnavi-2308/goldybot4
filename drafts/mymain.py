load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(os.path.dirname(__file__))

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
class SessionState(TypedDict):
    messages: Annotated[List, add_messages]
    session_id: str
    # user_id: str
    user_profile: Dict[str, Any]
    next_action: Optional[str]

## long term memory - user
class UserProfile(TypedDict):
    """User profile for personalization"""
    student_id: Optional[str] = None
    student_type: Optional[str] = None  # prospective, new, continuing
    interests: List[str] = field(default_factory=list)
    communication_style: str = "balanced"  # brief, detailed, balanced
    language: str = "english"
    completed_topics: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

## long term memory - knowledge base
class KnowledgeBase:
    """Simple persistent vector store using Chroma"""
    
    def __init__(self, db_path: str = "./knowledge_base.db"):
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



