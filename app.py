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
        self.document_loaded = False
        self.document_name = None
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
        
        try:
            # Get similarity scores from FAISS search
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
            
            if not docs_with_scores:
                return False
            
            # Check if best match exceeds threshold (lower score = higher similarity in FAISS)
            best_score = docs_with_scores[0][1]
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
    else:
        st.warning("⚠️ No document loaded. Please upload a document to start asking questions.")
    
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
                    if "cannot answer this question" in response.lower():
                        st.error(f"❌ **Response:** {response}")
                    else:
                        st.success(f"✅ **Response:** {response}")
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
