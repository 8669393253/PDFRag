# PDFRag
This project allows users to interact with PDF documents by asking questions and receiving context-aware answers. The system extracts text from PDFs, embeds the content using a transformer model, stores and retrieves embeddings using FAISS, and generates responses using the Groq API with LLaMA 3 models.

## Features
- **Upload and Process PDFs**: Users can upload a PDF, and the text will be extracted for further processing.
- **Efficient Text Chunking**: The extracted text is split into manageable chunks while preserving context.
- **Embeddings and Vector Storage**: Sentences are embedded using a transformer model and stored in a FAISS index.
- **Query Processing**: Users can enter queries, and the system retrieves the most relevant content from the PDF.
- **AI-Powered Responses**: Uses LLaMA 3 via the Groq API to generate human-like answers.
- **Embeddings Caching**: If embeddings exist for a document, they are loaded to improve efficiency.

## How It Works
1. **PDF Upload**: Users upload a PDF file through the Streamlit interface.
2. **Text Extraction & Cleaning**: The system extracts text using PyPDF2 and applies preprocessing.
3. **Text Splitting**: The document is split into overlapping chunks to improve contextual retrieval.
4. **Embedding Generation**: Each chunk is embedded using a Sentence Transformer model.
5. **FAISS Indexing**: The embeddings are stored in a FAISS index to enable fast similarity searches.
6. **Query Input**: Users enter questions related to the document.
7. **Similarity Search**: FAISS retrieves the most relevant document chunks based on query embeddings.
8. **Response Generation**: The retrieved content is sent to LLaMA 3, which generates a relevant response.
9. **Displaying Results**: The answer is displayed, and users can refine their queries.

## Prerequisites
- **Python 3.8+**: Ensure you have a compatible Python version installed.
- **Streamlit**: For the web interface.
- **FAISS**: For efficient similarity search.
- **Sentence Transformers**: For embedding generation.
- **PyPDF2**: For extracting text from PDFs.
- **Groq API Key**: Required to access LLaMA 3 models.

## Installation
1. Install dependencies using pip.
2. Obtain a Groq API key and store it in a `config.json` file.
3. Run the Streamlit app and start interacting with PDFs.

## Performance Considerations
- **Embedding Storage**: To improve efficiency, embeddings are saved locally and reused if the same document is uploaded again.
- **FAISS Optimization**: Uses an inner-product FAISS index for fast retrieval.
- **Token Limitations**: The system manages token limits to avoid exceeding LLaMA 3’s maximum context size.

## Possible Improvements
- Support for multiple PDFs in a single session.
- Reranking retrieved chunks for improved accuracy.
- Alternative embedding models for better vector representations.
- Integration with different LLMs for flexible response generation.

## Conclusion
This project provides an interactive way to explore PDFs using AI-driven search and response generation. By leveraging FAISS and LLaMA 3, it enables efficient and intelligent document querying.
