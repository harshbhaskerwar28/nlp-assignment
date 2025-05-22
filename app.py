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
import PyPDF2
import docx
from io import BytesIO

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

class DocumentProcessor:
    """
    Document processor to handle different file formats
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    @staticmethod
    def extract_text_from_docx(docx_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(BytesIO(docx_file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return None
    
    @staticmethod
    def extract_text_from_txt(txt_file):
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8').strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            return None

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
    
    def __init__(self, relevance_threshold=0.5):
        """
        Initialize RAG Pipeline
        
        Args:
            relevance_threshold (float): Minimum similarity score for relevant documents
        """
        self.vector_store = None
        self.relevance_threshold = relevance_threshold
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        self.document_loaded = False
        self.document_name = None
        self.document_summary = None
        logger.info(f"RAG Pipeline initialized with threshold: {relevance_threshold}")
    
    def load_knowledge_base(self, document_text, document_name="Unknown Document"):
        """
        Load and process document text into vector store
        
        Args:
            document_text (str): Raw text content of the knowledge base document
            document_name (str): Name of the document for reference
        """
        try:
            if not document_text or len(document_text.strip()) < 50:
                logger.error("Document text is too short or empty")
                return False
                
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
            self.document_loaded = True
            self.document_name = document_name
            self.document_summary = None  # Initialize as None, will generate on demand
            
            logger.info("Vector store created successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def is_general_document_question(self, query):
        """
        Check if the query is asking about the document in general
        """
        general_phrases = [
            "what is this document about", 
            "what is the document about",
            "what is this pdf about", 
            "what is the pdf about",
            "what's in this document",
            "what does this document contain",
            "what is the topic of",
            "summarize the document",
            "document summary"
        ]
        query_lower = query.lower()
        
        return any(phrase in query_lower for phrase in general_phrases)
    
    def check_relevance(self, query, retrieved_docs):
        """
        Check if retrieved documents are relevant to the query
        
        Args:
            query (str): User query
            retrieved_docs: Documents retrieved from vector search
            
        Returns:
            bool: True if relevant, False otherwise
        """
        # Always consider general document questions as relevant
        if hasattr(self, 'is_general_document_question') and self.is_general_document_question(query):
            logger.info("Query identified as a general document question")
            return True
            
        if not retrieved_docs:
            return False
        
        try:
            # Get similarity scores from FAISS search
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
            
            if not docs_with_scores:
                return False
            
            # Check if best match exceeds threshold (lower score = higher similarity in FAISS)
            best_score = min(score for _, score in docs_with_scores)  # Get best score from all results
            # Use a more robust conversion from distance to similarity
            relevance_score = 1 / (1 + best_score)  # Convert distance to similarity
            
            logger.info(f"Relevance score: {relevance_score:.3f}, Threshold: {self.relevance_threshold}")
            return relevance_score >= self.relevance_threshold
        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            return False
    
    def generate_response(self, query):
        """
        Generate response using RAG pipeline
        
        Args:
            query (str): User question
            
        Returns:
            str: Generated response or fallback message
        """
        if not self.vector_store or not self.document_loaded:
            return "Please load a knowledge base document first."
        
        try:
            # Handle general document questions directly
            if self.is_general_document_question(query):
                if hasattr(self, 'document_summary') and self.document_summary:
                    return f"This document is about: {self.document_summary}"
                else:
                    return f"This is a document named '{self.document_name}'. To get specific information, please ask more specific questions about its content."
            
            # Step 1: Retrieve relevant documents using similarity search
            retrieved_docs = self.vector_store.similarity_search(query, k=4)  # Increased from 3 to 4
            
            # Step 2: Check relevance using threshold
            if not self.check_relevance(query, retrieved_docs):
                return f"I don't have enough information in the document to answer this question confidently. Please try asking a different question about '{self.document_name}'."
            
            # Step 3: Generate response using LLM with retrieved context
            prompt_template = """
            You are an AI assistant that answers questions based on the provided context from a document.
            Answer the question based only on the provided context. Be concise and accurate.
            If the context doesn't contain enough information to provide a complete answer, 
            use what is available to give the best possible response.
            
            Document name: {document_name}
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "document_name"]
            )
            
            # Create QA chain
            qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            # Generate response
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            response = qa_chain(
                {"input_documents": retrieved_docs, "question": query, "document_name": self.document_name},
                return_only_outputs=True
            )
            
            logger.info("Response generated successfully")
            return response['output_text']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Talk to PDF",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG pipeline in session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    # CSS for chat interface
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding-left: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("📄 Talk to PDF")
        
        # Document Upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file:
            # Process the uploaded file
            with st.spinner("Reading document..."):
                document_text = None
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'txt':
                    document_text = DocumentProcessor.extract_text_from_txt(uploaded_file)
                elif file_extension == 'pdf':
                    document_text = DocumentProcessor.extract_text_from_pdf(uploaded_file)
                elif file_extension == 'docx':
                    document_text = DocumentProcessor.extract_text_from_docx(uploaded_file)
                
                if document_text:
                    # Load the document into the RAG pipeline
                    with st.spinner("Processing document..."):
                        if st.session_state.rag_pipeline.load_knowledge_base(document_text, uploaded_file.name):
                            st.success(f"Document loaded: {uploaded_file.name}")
                            st.session_state.chat_history = []  # Clear chat history on new document
                        else:
                            st.error("Failed to process document")
                else:
                    st.error("Could not read document")
        
        # Document info
        if st.session_state.rag_pipeline.document_loaded:
            st.info(f"Active document: {st.session_state.rag_pipeline.document_name}")
            
            # Optional document summary button
            if st.button("Generate Document Summary"):
                with st.spinner("Creating summary..."):
                    try:
                        document_preview = st.session_state.rag_pipeline.vector_store.similarity_search("document summary", k=5)
                        text_preview = "\n".join([doc.page_content for doc in document_preview])
                        summary_prompt = f"Create a short summary of this document: {text_preview}"
                        summary = st.session_state.rag_pipeline.llm.invoke(summary_prompt).content
                        st.session_state.rag_pipeline.document_summary = summary
                        st.success("Summary created!")
                    except Exception as e:
                        st.error(f"Error creating summary")
        
        # Advanced settings (collapsed by default)
        with st.expander("Advanced Settings", expanded=False):
            threshold = st.slider(
                "Relevance Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            st.session_state.rag_pipeline.relevance_threshold = threshold
    
    # Main chat interface
    if not st.session_state.rag_pipeline.document_loaded:
        # Welcome screen when no document is loaded
        st.markdown(
            """
            <div style='text-align: center; margin-top: 3rem'>
                <h1>👋 Welcome to Talk to PDF</h1>
                <p>Upload a document in the sidebar to get started</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Chat interface when document is loaded
        st.header("💬 Chat with your Document")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message">❓ {message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                content = message["content"]
                is_error = "don't have enough information" in content.lower() or "cannot answer" in content.lower()
                icon = "❌" if is_error else "✅"
                container_style = "error-container" if is_error else ""
                
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="message">{icon} {content}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_input("Ask a question:", key="user_query")
        
        # Generate response when user submits a question
        if user_query:
            # Add user query to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate and display response
            with st.spinner("Thinking..."):
                response = st.session_state.rag_pipeline.generate_response(user_query)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.rerun()
            
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
