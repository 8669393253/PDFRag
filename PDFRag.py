import streamlit as st
import os
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import re

# Load API key from config.json
with open("config.json") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedding_model.encode(text, normalize_embeddings=True)

def estimate_tokens(text):
    # More accurate token estimation (approx 1 token = 4 characters)
    return max(len(text) // 4, 1)  # Ensure at least 1 token

def clean_text(text):
    # Remove excessive whitespace and non-printable characters
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def main():
    st.header("Chat with PDF ☁")

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        st.write(f"Processing: {pdf.name}")

        # PDF processing with error handling
        try:
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            text = clean_text(text)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return

        # Text splitting with overlap for context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # ~500 tokens
            chunk_overlap=300,  # Preserve context between chunks
            length_function=estimate_tokens
        )
        sub_chunks = text_splitter.split_text(text)
        st.write(f"Split into {len(sub_chunks)} contextual chunks")

        # Embedding and storage handling
        store_name = pdf.name[:-4]
        vector_store = None
        
        if os.path.exists(f"{store_name}.pkl"):
            try:
                with open(f"{store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
                st.success('Loaded precomputed embeddings')
            except Exception as e:
                st.warning(f"Error loading embeddings: {str(e)}")

        if not vector_store:
            try:
                with st.spinner('Creating embeddings (this may take a minute)...'):
                    embeddings = np.array([get_embedding(chunk) for chunk in sub_chunks])
                    index = faiss.IndexFlatIP(384)  # Inner product for cosine similarity
                    index.add(embeddings.astype(np.float32))
                    vector_store = {"index": index, "chunks": sub_chunks}
                
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)
                st.success('Created and stored new embeddings')
            except Exception as e:
                st.error(f"Embedding creation failed: {str(e)}")
                return

        # Query handling
        query = st.text_input("Ask a question about your PDF:")
        if query:
            with st.spinner('Analyzing document...'):
                try:
                    # Get relevant chunks using FAISS
                    query_embedding = get_embedding(clean_text(query))
                    D, I = vector_store["index"].search(
                        np.array([query_embedding]).astype(np.float32), 
                        k=min(5, len(sub_chunks))  # Ensure we don't request more than available
                    )

                    valid_indices = [i for i in I[0] if i < len(sub_chunks)]
                    if not valid_indices:
                        st.warning("No matching content found in document")
                        return

                    # Get top chunks with scores
                    chunk_scores = [(sub_chunks[i], D[0][j]) 
                                  for j, i in enumerate(valid_indices)]
                    chunk_scores.sort(key=lambda x: x[1], reverse=True)

                    # Build context with token limit
                    context = []
                    total_tokens = 0
                    max_context_tokens = 3000  # Conservative limit for 8k model

                    for chunk, score in chunk_scores:
                        chunk_tokens = estimate_tokens(chunk)
                        if total_tokens + chunk_tokens <= max_context_tokens:
                            context.append(chunk)
                            total_tokens += chunk_tokens
                        else:
                            remaining = max_context_tokens - total_tokens
                            if remaining > 100:  # Only add partial if meaningful
                                context.append(chunk[:remaining*4])  # Approx chars
                            break

                    if not context:
                        st.warning("Insufficient relevant content found")
                        return

                    full_context = "\n".join(context)
                    # st.subheader("Most Relevant Content Found")
                    # st.write(full_context[:2000] + "...")  # Show preview

                    # Generate answer
                    with st.spinner('Generating answer...'):
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{
                                "role": "system",
                                "content": f"Answer based on this context: {full_context}"
                            }, {
                                "role": "user",
                                "content": query
                            }],
                            temperature=0.4,
                            max_tokens=1024
                        )
                        
                        answer = response.choices[0].message.content
                        st.subheader("Answer")
                        st.write(answer)

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == '__main__':
    main()