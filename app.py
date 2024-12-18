import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    SimpleDirectoryReader
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# OpenAI import
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_files(uploaded_files):
    """
    Load documents from uploaded files
    """
    documents = []
    
    # Create a temporary directory to save uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temp directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Use SimpleDirectoryReader to load files from temp directory
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
    
    return documents

def create_index(documents):
    """
    Create a vector index from documents
    """
    if not documents:
        st.error("No documents to index!")
        return None
    
    try:
        # Create index from documents
        index = VectorStoreIndex.from_documents(documents)
        return index
    
    except Exception as e:
        st.error(f"Error creating index: {e}")
        logger.error(f"Index creation error: {e}")
        return None

def create_query_engine(index):
    """
    Create query engine with retriever and postprocessor
    """
    if index is None:
        st.error("Cannot create query engine. Index is None.")
        return None
    
    try:
        # Configure OpenAI LLM
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        
        retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor]
        )
        
        return query_engine
    
    except Exception as e:
        st.error(f"Error creating query engine: {e}")
        logger.error(f"Query engine creation error: {e}")
        return None

def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    
    # Create two columns
    left_column, right_column = st.columns([1, 2])
    
    with left_column:
        st.title("RAG Multi Document Query Assistant")
        
        # API Key Input
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        # File Upload
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md']
        )
        
        # Process Files Button
        process_button = st.button("Process Documents")
        
        # Query Input (initially disabled)
        query = st.text_input("Enter your query:")
    
    with right_column:
        # Display area for results
        if api_key:
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = api_key
            
            if uploaded_files and process_button:
                # Load documents from uploaded files
                with st.spinner('Loading and indexing documents...'):
                    try:
                        # Load documents
                        documents = load_documents_from_files(uploaded_files)
                        
                        # Create index
                        index = create_index(documents)
                        
                        if index is None:
                            st.error("Failed to create index.")
                            return
                        
                        # Create query engine
                        query_engine = create_query_engine(index)
                        
                        if query_engine is None:
                            st.error("Failed to create query engine.")
                            return
                        
                        # Store query engine in session state
                        st.session_state.query_engine = query_engine
                        
                        # Success message
                        st.success(f"Processed {len(documents)} documents successfully!")
                    
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
            
            # Query Processing
            if query and 'query_engine' in st.session_state:
                with st.spinner('Processing query...'):
                    try:
                        # Perform query
                        response = st.session_state.query_engine.query(query)
                        
                        # Display Response
                        st.subheader("Response:")
                        st.write(str(response))
                        
                        # Display Source Nodes
                        st.subheader("Source Nodes:")
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            for node in response.source_nodes:
                                with st.expander(f"Source - Similarity: {node.score:.2f}"):
                                    st.write(node.text)
                        else:
                            st.info("No source nodes found.")
                    
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        else:
            st.info("Please enter your OpenAI API Key to proceed.")

if __name__ == "__main__":
    main()