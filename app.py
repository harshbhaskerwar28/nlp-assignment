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
        page_title="RAG Pipeline Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Pipeline Implementation")
    st.subheader("Document-based Q&A with Relevance Filtering")
    
    # Initialize RAG pipeline in session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    # Document Upload Section
    st.header("📄 Upload Knowledge Base Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['txt', 'pdf', 'docx'],
        help="Upload a document to use as your knowledge base. Supported formats: TXT, PDF, DOCX"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing document..."):
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
                if st.session_state.rag_pipeline.load_knowledge_base(document_text, uploaded_file.name):
                    st.success(f"✅ Document '{uploaded_file.name}' loaded successfully!")
                    
                    # Display document preview
                    with st.expander("📖 Document Preview"):
                        preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
                        st.text_area("Document Content", preview_text, height=200, disabled=True)
                        st.info(f"Document length: {len(document_text)} characters")
                else:
                    st.error("❌ Failed to load the document. Please check if the document contains enough text.")
            else:
                st.error("❌ Failed to extract text from the document. Please try a different file.")
    
    # Show current document status
    if st.session_state.rag_pipeline.document_loaded:
        st.info(f"📚 Current Knowledge Base: {st.session_state.rag_pipeline.document_name}")
        
        # Replace automatic summary display with a button to show summary on demand
        if st.button("📝 Generate Document Summary"):
            with st.spinner("Generating summary..."):
                if not hasattr(st.session_state.rag_pipeline, 'document_summary') or not st.session_state.rag_pipeline.document_summary:
                    try:
                        # Generate summary if it doesn't exist
                        if hasattr(st.session_state.rag_pipeline, 'llm'):
                            document_preview = st.session_state.rag_pipeline.vector_store.similarity_search("document summary", k=5)
                            text_preview = "\n".join([doc.page_content for doc in document_preview])
                            summary_prompt = f"""
                            Create a short summary (2-3 sentences) of this document that describes what it's about:
                            
                            {text_preview}
                            
                            Summary:
                            """
                            st.session_state.rag_pipeline.document_summary = st.session_state.rag_pipeline.llm.invoke(summary_prompt).content
                        else:
                            st.session_state.rag_pipeline.document_summary = f"A document named {st.session_state.rag_pipeline.document_name}"
                    except Exception as e:
                        logger.error(f"Error generating summary: {str(e)}")
                        st.session_state.rag_pipeline.document_summary = f"A document named {st.session_state.rag_pipeline.document_name}"
                
                st.success(f"Document Summary: {st.session_state.rag_pipeline.document_summary}")
    else:
        st.warning("⚠️ No document loaded. Please upload a document to start asking questions.")
    
    # Configuration section
    st.sidebar.header("⚙️ Configuration")
    threshold = st.sidebar.slider(
        "Relevance Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher threshold = stricter relevance filtering"
    )
    st.session_state.rag_pipeline.relevance_threshold = threshold
    
    # Instructions section
    st.sidebar.header("📋 Instructions")
    st.sidebar.write("""
    1. **Upload Document**: Choose a TXT, PDF, or DOCX file
    2. **Wait for Processing**: Document will be chunked and embedded
    3. **Ask Questions**: Enter questions related to your document
    4. **Adjust Threshold**: Control relevance filtering strictness
    """)
    
    st.sidebar.header("💡 Tips")
    st.sidebar.write("""
    - **Relevant queries** will be answered based on your document
    - **Irrelevant queries** will be rejected with a message
    - Higher threshold = stricter filtering
    - Lower threshold = more lenient filtering
    """)
    
    # Main chat interface (only show if document is loaded)
    if st.session_state.rag_pipeline.document_loaded:
        st.header("💬 Ask Questions About Your Document")
        
        # Query input
        user_query = st.text_input(
            "Enter your question:",
            placeholder="Ask something about your uploaded document..."
        )
        
        # Generate response button
        if st.button("🔍 Get Answer", type="primary"):
            if user_query.strip():
                with st.spinner("Processing your question..."):
                    response = st.session_state.rag_pipeline.generate_response(user_query)
                    
                    # Display response with styling
                    if "don't have enough information" in response.lower() or "cannot answer this question" in response.lower():
                        st.error(f"{response}")
                    else:
                        st.success(f"✅ Response: {response}")
            else:
                st.warning("Please enter a question.")
        
        # Chat history (optional enhancement)
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        if st.session_state.chat_history:
            st.header("💭 Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                with st.expander(f"Q{i+1}: {q[:50]}..."):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")
    
    # Display pipeline information
    st.header("🔧 Pipeline Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **RAG Pipeline Steps:**
        1. Document upload & text extraction
        2. Document chunking & embedding
        3. Vector storage in FAISS
        4. Query similarity search
        5. Relevance threshold check
        6. LLM response generation
        """)
    
    with col2:
        st.info(f"""
        **Current Configuration:**
        - Model: Gemini 2.0 Flash
        - Embedding: Google text-embedding-001
        - Relevance Threshold: {threshold}
        - Vector Store: FAISS
        - Document Loaded: {'✅' if st.session_state.rag_pipeline.document_loaded else '❌'}
        """)

if __name__ == "__main__":
    main()
