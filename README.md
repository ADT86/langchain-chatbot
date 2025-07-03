# 📚 PDF Chatbot with LangChain

A simple and powerful chatbot that can read PDF files and answer questions about their content using LangChain and OpenAI.

## Features

- 📄 **PDF Processing**: Load and process PDF documents
- 🤖 **AI-Powered Chat**: Ask questions about your PDF content
- 💾 **Memory**: Maintains conversation context
- 🔍 **Source Tracking**: See which parts of the PDF were used to answer questions
- 🌐 **Web Interface**: Beautiful Streamlit web app
- 💻 **Command Line**: Simple CLI version for terminal users

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Web Interface (Recommended)

Run the Streamlit web app:

```bash
streamlit run pdf_chatbot.py
```

Then:
1. Open your browser to the provided URL (usually http://localhost:8501)
2. Upload your PDF file
3. Start asking questions!

### Command Line Interface

Use the CLI version for a terminal-based experience:

```bash
python cli_chatbot.py path/to/your/document.pdf
```

Example:
```bash
python cli_chatbot.py research_paper.pdf
```

## How It Works

1. **PDF Loading**: The chatbot uses PyPDF to extract text from your PDF
2. **Text Chunking**: Long documents are split into smaller, manageable chunks
3. **Vector Embeddings**: Text chunks are converted to vector embeddings using OpenAI
4. **Vector Store**: Chunks are stored in a FAISS vector database for fast retrieval
5. **Question Answering**: When you ask a question:
   - The question is converted to an embedding
   - Similar chunks are retrieved from the vector store
   - The relevant context is sent to GPT-3.5-turbo
   - You get an answer based on your PDF content

## Example Questions

Once you've uploaded a PDF, you can ask questions like:

- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the conclusions?"
- "Explain the methodology used"
- "What data was collected?"
- "What are the limitations mentioned?"

## Configuration

You can customize the chatbot by modifying these parameters in the code:

- **Chunk Size**: How large each text chunk should be (default: 1000 characters)
- **Chunk Overlap**: Overlap between chunks for better context (default: 200 characters)
- **Model**: Which OpenAI model to use (default: gpt-3.5-turbo)
- **Temperature**: How creative the responses should be (default: 0.7)

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Make sure you've created a `.env` file with your API key
   - Check that the key is valid and has sufficient credits

2. **"Error loading PDF"**
   - Ensure the PDF file is not corrupted
   - Check that the file path is correct
   - Some PDFs with images only may not work well

3. **"Error creating vector store"**
   - Check your internet connection
   - Verify your OpenAI API key is working

### Performance Tips

- For large PDFs (>50 pages), processing may take a few minutes
- The first question after uploading may be slower as the system initializes
- Consider using smaller chunk sizes for more precise answers

## Dependencies

- `langchain`: Core LangChain functionality
- `langchain-community`: Community integrations
- `langchain-openai`: OpenAI integration
- `pypdf`: PDF processing
- `streamlit`: Web interface
- `python-dotenv`: Environment variable management
- `faiss-cpu`: Vector storage (included with langchain-community)

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this chatbot! 