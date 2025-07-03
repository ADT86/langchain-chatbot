#!/usr/bin/env python3
"""
Test script to verify PDF chatbot setup
Run this to check if everything is working correctly
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import langchain
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain: {e}")
        return False
    
    try:
        import langchain_community
        print("✅ langchain_community imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain_community: {e}")
        return False
    
    try:
        import langchain_openai
        print("✅ langchain_openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain_openai: {e}")
        return False
    
    try:
        import pypdf
        print("✅ pypdf imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pypdf: {e}")
        return False
    
    try:
        import streamlit
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import streamlit: {e}")
        return False
    
    try:
        import dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import python-dotenv: {e}")
        return False
    
    return True

def test_openai_key():
    """Test if OpenAI API key is set and valid"""
    print("\n🔑 Testing OpenAI API key...")
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("   Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ Please replace the placeholder API key with your actual OpenAI API key")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ API key format looks incorrect. Should start with 'sk-'")
        return False
    
    print("✅ OpenAI API key found and format looks correct")
    return True

def test_openai_connection():
    """Test if we can connect to OpenAI API"""
    print("\n🌐 Testing OpenAI connection...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=50
        )
        
        # Simple test query
        response = llm.invoke("Say 'Hello, world!' in one word.")
        print("✅ Successfully connected to OpenAI API")
        print(f"   Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to OpenAI API: {e}")
        print("   Please check your API key and internet connection")
        return False

def main():
    print("=" * 50)
    print("🧪 PDF Chatbot Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Test API key
    if not test_openai_key():
        print("\n❌ API key test failed. Please set up your .env file.")
        sys.exit(1)
    
    # Test OpenAI connection
    if not test_openai_connection():
        print("\n❌ OpenAI connection test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your setup is ready.")
    print("=" * 50)
    print("\nYou can now run:")
    print("  • Web interface: streamlit run pdf_chatbot.py")
    print("  • CLI version: python cli_chatbot.py your_file.pdf")

if __name__ == "__main__":
    main() 