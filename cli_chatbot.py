#!/usr/bin/env python3
"""
Command-line PDF Chatbot using LangChain
Usage: python cli_chatbot.py <path_to_pdf>
"""

import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()
    print("🔧 Environment variables loaded from .env file")
except Exception as e:
    print(f"⚠️  Warning: Could not load .env file: {e}")
    print("   The script will still work if OPENAI_API_KEY is set as an environment variable")

def load_pdf(file_path):
    """Load and process PDF file"""
    try:
        print(f"📖 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ PDF processed successfully! ({len(chunks)} chunks created)")
        return chunks
    except Exception as e:
        print(f"❌ Error loading PDF: {str(e)}")
        return None

def create_vectorstore(chunks):
    """Create vector store from document chunks"""
    try:
        print("🔍 Creating vector store...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

def create_conversation_chain(vectorstore):
    """Create conversation chain with memory"""
    try:
        print("🤖 Setting up chatbot...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )
        
        print("✅ Chatbot ready!")
        return conversation_chain
    except Exception as e:
        print(f"❌ Error creating conversation chain: {str(e)}")
        return None

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python cli_chatbot.py <path_to_pdf>")
        print("Example: python cli_chatbot.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please ensure your API key is set in one of these ways:")
        print("1. Create a .env file in this directory with: OPENAI_API_KEY=your_api_key_here")
        print("2. Set it as an environment variable: export OPENAI_API_KEY=your_api_key_here")
        print("3. Check if the .env file exists and contains the correct key")
        
        # Check if .env file exists
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if os.path.exists(env_file):
            print(f"📁 Found .env file at: {env_file}")
            print("   Make sure it contains: OPENAI_API_KEY=your_api_key_here")
        else:
            print(f"📁 No .env file found at: {env_file}")
            print("   You can copy env_example.txt to .env and update it with your API key")
        sys.exit(1)
    
    # Validate API key format (basic check)
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: OpenAI API key should start with 'sk-'")
        print("   Please verify your API key is correct")
    
    print(f"✅ OpenAI API key loaded (ends with: ...{api_key[-4:]})")
    
    print("=" * 50)
    print("📚 PDF Chatbot - Command Line Interface")
    print("=" * 50)
    
    # Load and process PDF
    chunks = load_pdf(pdf_path)
    if not chunks:
        sys.exit(1)
    
    # Create vector store
    vectorstore = create_vectorstore(chunks)
    if not vectorstore:
        sys.exit(1)
    
    # Create conversation chain
    conversation = create_conversation_chain(vectorstore)
    if not conversation:
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("💬 Chat with your PDF! (Type 'quit' to exit)")
    print("=" * 50)
    
    # Chat loop
    while True:
        try:
            # Get user input
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Get response
            print("🤖 Thinking...")
            response = conversation({"question": question})
            answer = response["answer"]
            
            print(f"\n💡 Answer: {answer}")
            
            # Show source information
            if "source_documents" in response and response["source_documents"]:
                print(f"\n📄 Sources: {len(response['source_documents'])} relevant chunks found")
                for i, doc in enumerate(response["source_documents"][:2]):
                    page = doc.metadata.get('page', 'Unknown')
                    print(f"   Source {i+1}: Page {page}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 