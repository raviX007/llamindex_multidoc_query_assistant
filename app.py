import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_files(uploaded_files):
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
    return documents

def create_index(documents):
    if not documents:
        st.error("No documents to index!")
        return None
   
    try:
        index = VectorStoreIndex.from_documents(documents)
        return index
   
    except Exception as e:
        st.error(f"Error creating index: {e}")
        logger.error(f"Index creation error: {e}")
        return None

def create_query_engine(index):
    if index is None:
        st.error("Cannot create query engine. Index is None.")
        return None
   
    try:
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
    st.set_page_config(layout="wide")
    left_column, right_column = st.columns([1, 2])
   
    with left_column:
        st.title("LlamaIndex Multi-Document RAG")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md']
        )
        process_button = st.button("Process Documents")
        query = st.text_input("Enter your query:")
   
    with right_column:
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
           
            if uploaded_files and process_button:
                with st.spinner('Loading and indexing documents...'):
                    try:
                        documents = load_documents_from_files(uploaded_files)
                        index = create_index(documents)
                       
                        if index is None:
                            st.error("Failed to create index.")
                            return
                       
                        query_engine = create_query_engine(index)
                       
                        if query_engine is None:
                            st.error("Failed to create query engine.")
                            return
                       
                        st.session_state.query_engine = query_engine
                        st.success(f"Processed {len(documents)} documents successfully!")
                   
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
           
            if query and 'query_engine' in st.session_state:
                with st.spinner('Processing query...'):
                    try:
                        response = st.session_state.query_engine.query(query)
                        st.subheader("Response:")
                        st.write(str(response))
                       
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
