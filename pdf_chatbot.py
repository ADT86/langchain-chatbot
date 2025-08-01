import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for deployed apps)
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except:
        pass
    
    # Fall back to environment variables (for local development)
    return os.getenv("OPENAI_API_KEY")

def load_pdf(file_path):
    """Load and process PDF file"""
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

def create_vectorstore(chunks):
    """Create vector store from document chunks"""
    try:
        # Use the unified API key function
        api_key = get_openai_api_key()
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_conversation_chain(vectorstore):
    """Create conversation chain with memory"""
    try:
        # Use the unified API key function
        api_key = get_openai_api_key()
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=api_key
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify which key to store in memory
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot")
    st.markdown("Upload a PDF file and ask questions about its content!")
    
    # Check if OpenAI API key is set
    api_key = get_openai_api_key()
    if not api_key:
        st.error("⚠️ OpenAI API key not found!")
        st.info("Please set your OpenAI API key in one of these ways:")
        st.markdown("""
        **For local development:**
        - Create a `.env` file with: `OPENAI_API_KEY=your_api_key_here`
        
        **For Streamlit Cloud deployment:**
        - Go to your app settings → Secrets
        - Add: `OPENAI_API_KEY = "your_api_key_here"`
        
        **For GitHub Actions/Codespaces:**
        - Set as repository secret: `OPENAI_API_KEY`
        - Or set as environment variable
        """)
        return
    
    # Validate API key format
    if not api_key.startswith('sk-'):
        st.warning("⚠️ Warning: OpenAI API key should start with 'sk-'")
    else:
        st.success(f"✅ OpenAI API key loaded (ends with: ...{api_key[-4:]})")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to chat about its content"
    )
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process PDF
        with st.spinner("Processing PDF..."):
            chunks = load_pdf("temp_pdf.pdf")
            
            if chunks:
                st.success(f"✅ PDF loaded successfully! ({len(chunks)} chunks created)")
                
                # Create vector store
                with st.spinner("Creating vector store..."):
                    vectorstore = create_vectorstore(chunks)
                
                if vectorstore:
                    # Create conversation chain
                    with st.spinner("Setting up chatbot..."):
                        st.session_state.conversation = create_conversation_chain(vectorstore)
                    
                    if st.session_state.conversation:
                        st.success("🤖 Chatbot ready! You can now ask questions.")
                        
                        # Clean up temporary file
                        os.remove("temp_pdf.pdf")
                    else:
                        st.error("Failed to create conversation chain.")
                else:
                    st.error("Failed to create vector store.")
            else:
                st.error("Failed to load PDF.")
    
    # Chat interface
    if st.session_state.conversation:
        st.markdown("---")
        st.subheader("💬 Chat with your PDF")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation({"question": prompt})
                        answer = response["answer"]
                        
                        # Display answer
                        st.write(answer)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Show source documents if available
                        if "source_documents" in response and response["source_documents"]:
                            with st.expander("📄 View Sources"):
                                for i, doc in enumerate(response["source_documents"][:3]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content[:200] + "...")
                                    st.markdown(f"*Page: {doc.metadata.get('page', 'Unknown')}*")
                                    st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main() 