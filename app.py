import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and configure API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not api_key:
    st.error("GOOGLE_API_KEY environment variable not found. Please set your API key.")
    st.stop()

try:
    gen_ai.configure(api_key=api_key)
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {str(e)}")
    st.stop()

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) Pipeline for Document-based Q&A
    
    This pipeline implements the approach discussed in the assignment:
    1. Document ingestion and chunking
    2. Vector embedding and storage in FAISS
    3. Query similarity search with relevance threshold
    4. LLM-based response generation with context
    5. Fallback response for irrelevant queries
    """
    
    def __init__(self, relevance_threshold=0.7):
        """
        Initialize RAG Pipeline
        
        Args:
            relevance_threshold (float): Minimum similarity score for relevant documents
        """
        self.vector_store = None
        self.relevance_threshold = relevance_threshold
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        logger.info(f"RAG Pipeline initialized with threshold: {relevance_threshold}")
    
    def load_knowledge_base(self, document_text):
        """
        Load and process document text into vector store
        
        Args:
            document_text (str): Raw text content of the knowledge base document
        """
        try:
            # Step 1: Split document into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            text_chunks = text_splitter.split_text(document_text)
            logger.info(f"Document split into {len(text_chunks)} chunks")
            
            # Step 2: Create vector embeddings and store in FAISS
            self.vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            logger.info("Vector store created successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def check_relevance(self, query, retrieved_docs):
        """
        Check if retrieved documents are relevant to the query
        
        Args:
            query (str): User query
            retrieved_docs: Documents retrieved from vector search
            
        Returns:
            bool: True if relevant, False otherwise
        """
        if not retrieved_docs:
            return False
        
        # Get similarity scores from FAISS search
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
        
        if not docs_with_scores:
            return False
        
        # Check if best match exceeds threshold (lower score = higher similarity in FAISS)
        best_score = docs_with_scores[0][1]
        relevance_score = 1 / (1 + best_score)  # Convert distance to similarity
        
        logger.info(f"Relevance score: {relevance_score:.3f}, Threshold: {self.relevance_threshold}")
        return relevance_score >= self.relevance_threshold
    
    def generate_response(self, query):
        """
        Generate response using RAG pipeline
        
        Args:
            query (str): User question
            
        Returns:
            str: Generated response or fallback message
        """
        if not self.vector_store:
            return "Please load a knowledge base document first."
        
        try:
            # Step 1: Retrieve relevant documents using similarity search
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            
            # Step 2: Check relevance using threshold
            if not self.check_relevance(query, retrieved_docs):
                return "I cannot answer this question as it is not relevant to the loaded document."
            
            # Step 3: Generate response using LLM with retrieved context
            prompt_template = """
            You are an AI assistant that answers questions based only on the provided context.
            If the context doesn't contain information to answer the question, say that you cannot answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer (be concise and accurate):
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            # Generate response
            response = qa_chain(
                {"input_documents": retrieved_docs, "question": query},
                return_only_outputs=True
            )
            
            logger.info("Response generated successfully")
            return response['output_text']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

def load_sample_document():
    """
    Load sample document about Artificial Intelligence
    This serves as our knowledge base for demonstration
    """
    document_text = """
    Artificial Intelligence (AI) and Machine Learning
    
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. AI has become an integral part of the technology industry and is 
    revolutionizing various sectors including healthcare, finance, transportation, and education.
    
    Machine Learning is a subset of AI that provides systems the ability to automatically learn and 
    improve from experience without being explicitly programmed. Machine learning focuses on the 
    development of computer programs that can access data and use it to learn for themselves.
    
    Types of Machine Learning:
    1. Supervised Learning: Uses labeled training data to learn a mapping function from input to output
    2. Unsupervised Learning: Finds hidden patterns in data without labeled examples
    3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers to 
    model and understand complex patterns in data. It has been particularly successful in areas like 
    image recognition, natural language processing, and speech recognition.
    
    Applications of AI:
    - Healthcare: Medical diagnosis, drug discovery, personalized treatment
    - Finance: Fraud detection, algorithmic trading, risk assessment
    - Transportation: Autonomous vehicles, traffic optimization
    - Technology: Virtual assistants, recommendation systems, search engines
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers 
    and humans through natural language. NLP enables machines to read, understand, and derive meaning 
    from human language in a valuable way.
    
    Computer Vision is another important field of AI that trains computers to interpret and understand 
    the visual world. Using digital images from cameras and videos and deep learning models, machines 
    can accurately identify and classify objects.
    
    The future of AI holds immense potential with developments in quantum computing, edge AI, and 
    explainable AI making systems more powerful, efficient, and transparent.
    """
    return document_text

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="RAG Pipeline Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Pipeline Implementation")
    st.subheader("Document-based Q&A with Relevance Filtering")
    
    # Initialize RAG pipeline in session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
        # Load sample document
        sample_doc = load_sample_document()
        if st.session_state.rag_pipeline.load_knowledge_base(sample_doc):
            st.success("✅ Knowledge base loaded successfully!")
        else:
            st.error("❌ Failed to load knowledge base")
    
    # Display knowledge base info
    with st.expander("📄 View Knowledge Base Document"):
        st.write("**Sample Document: Artificial Intelligence and Machine Learning**")
        st.write(load_sample_document())
    
    # Configuration section
    st.sidebar.header("⚙️ Configuration")
    threshold = st.sidebar.slider(
        "Relevance Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher threshold = stricter relevance filtering"
    )
    st.session_state.rag_pipeline.relevance_threshold = threshold
    
    # Example queries section
    st.sidebar.header("💡 Example Queries")
    st.sidebar.write("**Relevant queries:**")
    st.sidebar.write("- What is machine learning?")
    st.sidebar.write("- Types of machine learning")
    st.sidebar.write("- Applications of AI in healthcare")
    
    st.sidebar.write("**Irrelevant queries:**")
    st.sidebar.write("- What is the weather today?")
    st.sidebar.write("- How to cook pasta?")
    st.sidebar.write("- Latest movie reviews")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Query input
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning?"
    )
    
    # Generate response button
    if st.button("🔍 Get Answer", type="primary"):
        if user_query.strip():
            with st.spinner("Processing your question..."):
                response = st.session_state.rag_pipeline.generate_response(user_query)
                
                # Display response with styling
                if "cannot answer this question" in response.lower():
                    st.error(f"❌ **Response:** {response}")
                else:
                    st.success(f"✅ **Response:** {response}")
        else:
            st.warning("Please enter a question.")
    
    # Display pipeline information
    st.header("🔧 Pipeline Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **RAG Pipeline Steps:**
        1. Document chunking & embedding
        2. Vector storage in FAISS
        3. Query similarity search
        4. Relevance threshold check
        5. LLM response generation
        """)
    
    with col2:
        st.info(f"""
        **Current Configuration:**
        - Model: Gemini 2.0 Flash
        - Embedding: Google text-embedding-001
        - Relevance Threshold: {threshold}
        - Vector Store: FAISS
        """)

if __name__ == "__main__":
    main()
