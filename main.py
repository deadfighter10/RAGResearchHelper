import os
import io # Needed for reading bytes
import PyPDF2
import streamlit as st
# from huggingface_hub import login # Still unused
from langchain.chains.conversation.base import ConversationChain # Keep for potential future use, but not used in basic invoke
from langchain.memory import ConversationBufferMemory # Keep for potential future use
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.tools import ArxivQueryRun # Replaced by arxiv library
import logging
import requests # For potential URL downloads in the future (and handling arxiv download errors)
import arxiv # New library for interacting with arXiv API
from urllib.parse import urlparse # For extracting filenames from URLs

# --- Configuration and Initialization ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Expensive Resources ---

@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer model and caches it."""
    logging.info("Loading embedding model 'all-MiniLM-L6-v2'...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        st.error(f"Error loading embedding model: {e}. Please check model availability and network connection.")
        return None

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    logging.info("Initializing ChromaDB client...")
    try:
        client = chromadb.PersistentClient(path="chroma_db")
        logging.info("ChromaDB client initialized.")
        return client
    except Exception as e:
        logging.error(f"Error initializing ChromaDB client: {e}")
        st.error(f"Error initializing ChromaDB client: {e}")
        return None

# Modified to accept model name
@st.cache_resource
def load_llm(model_name="gemma3:12b"):
    """Loads the specified Ollama LLM instance and caches it."""
    logging.info(f"Loading Ollama LLM model: {model_name}...")
    try:
        llm = Ollama(model=model_name, base_url="http://localhost:11434")
        # Test connection
        llm.invoke("Respond with 'OK'.")
        logging.info(f"Ollama LLM '{model_name}' loaded successfully.")
        return llm
    except Exception as e:
        logging.error(f"Error loading Ollama LLM '{model_name}': {e}. Ensure Ollama is running and the model is pulled.")
        st.error(f"Error loading Ollama LLM '{model_name}': {e}. Please ensure Ollama is running at http://localhost:11434 and the model '{model_name}' is pulled.")
        return None

# Removed get_arxiv_tool as we now use the arxiv library directly

# --- Core Logic Functions ---

def extract_text_from_pdf_bytes(pdf_bytes, source_identifier):
    """Extracts text from PDF content provided as bytes."""
    logging.info(f"Extracting text from PDF bytes for source: {source_identifier}")
    text = ""
    metadata = {"source": source_identifier}
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        # Try to extract metadata from PDF properties
        pdf_meta = reader.metadata
        if pdf_meta:
            metadata["title"] = pdf_meta.get('/Title', 'N/A').strip()
            metadata["author"] = pdf_meta.get('/Author', 'N/A').strip()
        logging.info(f"PDF has {len(reader.pages)} pages.")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logging.warning(f"No text extracted from page {i+1} in {source_identifier}")
            except Exception as page_e:
                logging.error(f"Error extracting text from page {i+1} in {source_identifier}: {page_e}")
                st.warning(f"Could not extract text from page {i+1} of {source_identifier}.")
        logging.info(f"Successfully extracted text from {source_identifier}")
        return text, metadata
    except Exception as e:
        logging.error(f"Failed to read PDF bytes for {source_identifier}: {e}")
        st.error(f"Error reading PDF content from {source_identifier}: {e}")
        return None, metadata # Return metadata even if text extraction fails


def extract_text_from_uploaded_pdfs(uploaded_files):
    """Extracts text from a list of uploaded PDF files."""
    extracted_data = [] # List of tuples: (text, metadata)
    logging.info(f"Starting PDF text extraction for {len(uploaded_files)} file(s).")
    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.getvalue()
            text, metadata = extract_text_from_pdf_bytes(pdf_bytes, uploaded_file.name)
            if text is not None:
                extracted_data.append((text, metadata))
        except Exception as e:
            logging.error(f"Error processing uploaded file {uploaded_file.name}: {e}")
            st.warning(f"Could not process file: {uploaded_file.name}. Skipping.")
    return extracted_data


def process_text_and_store(client, text_embedding_model, docs_with_metadata, collection_name="knowledge_base"):
    """Splits text, generates embeddings with metadata, and stores them."""
    if not docs_with_metadata:
        logging.warning("No documents provided for processing.")
        st.warning("No text found to process.")
        return None
    if not client or not text_embedding_model:
        logging.error("Chroma client or embedding model not available for processing.")
        st.error("System components (DB Client or Embedding Model) missing. Cannot process text.")
        return None

    logging.info(f"Processing text for collection '{collection_name}'...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, # Increased overlap slightly more
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_counter = 0

    for doc_text, base_metadata in docs_with_metadata:
        if not doc_text:
            logging.warning(f"Skipping document with no text: {base_metadata.get('source', 'Unknown Source')}")
            continue
        chunks = text_splitter.split_text(doc_text)
        logging.info(f"Split '{base_metadata.get('source', 'doc')}' into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{chunk_counter}"
            chunk_metadata = base_metadata.copy() # Start with base metadata
            chunk_metadata["chunk_id"] = chunk_counter # Add chunk index
            # Add page number estimation if possible (simple method)
            # Note: This is approximate. Better methods exist.
            chunk_metadata["approx_page"] = i * text_splitter._chunk_size // 1000 + 1 # Rough estimate

            all_chunks.append(chunk)
            all_metadatas.append(chunk_metadata)
            all_ids.append(chunk_id)
            chunk_counter += 1

    if not all_chunks:
        logging.warning("Text splitting resulted in zero chunks across all documents.")
        st.warning("Could not split the text into processable chunks.")
        return None

    logging.info(f"Total chunks to process: {len(all_chunks)}")

    try:
        logging.info(f"Attempting to delete existing collection: {collection_name}")
        try:
            client.delete_collection(name=collection_name)
            logging.info(f"Successfully deleted existing collection: {collection_name}")
        except Exception:
            logging.info(f"Collection {collection_name} does not exist or couldn't be deleted.")
            pass

        logging.info(f"Creating new collection: {collection_name}")
        collection = client.create_collection(name=collection_name)

        batch_size = 100 # Adjust batch size based on memory/performance
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]

            logging.info(f"Encoding batch {i//batch_size + 1}/{len(all_chunks)//batch_size + 1} ({len(batch_chunks)} chunks)...")
            try:
                batch_embeddings = text_embedding_model.encode(batch_chunks).tolist()
            except Exception as e:
                logging.error(f"Error encoding batch {i//batch_size + 1}: {e}")
                st.error(f"Failed to generate embeddings for a text batch.")
                continue # Skip this batch

            try:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_chunks
                )
                logging.info(f"Added batch {i//batch_size + 1} to collection '{collection_name}'.")
            except Exception as e:
                 logging.error(f"Error adding batch {i//batch_size + 1} to ChromaDB: {e}")
                 st.error(f"Failed to store text batch in the vector database.")

        logging.info(f"Successfully processed and stored {len(all_chunks)} chunks in collection '{collection_name}'.")
        return collection

    except Exception as e:
        logging.error(f"Failed to process and store text in ChromaDB: {e}")
        st.error(f"An error occurred during text processing and storage: {e}")
        return None

def semantic_search(collection, text_embedding_model, query, top_k=5):
    """Performs semantic search and retrieves documents and metadata."""
    if not collection or not text_embedding_model:
         logging.error("Collection or embedding model not available for search.")
         st.error("Cannot perform search: Vector database or embedding model is missing.")
         return None
    logging.info(f"Performing semantic search for query: '{query}' with top_k={top_k}")
    try:
        query_embedding = text_embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas'] # Include metadata
        )
        # Chroma returns lists of lists, even for a single query
        res_docs = results.get('documents', [[]])[0]
        res_metas = results.get('metadatas', [[]])[0]
        logging.info(f"Semantic search returned {len(res_docs)} results.")
        return res_docs, res_metas
    except Exception as e:
        logging.error(f"Error during semantic search: {e}")
        st.error(f"An error occurred during semantic search: {e}")
        return None, None

# --- Streamlit UI ---

def main():
    st.set_page_config(layout="wide")
    st.title("📄 Enhanced RAG Research Assistant")
    st.markdown("Upload PDFs or search arXiv (full PDF), manage chat, and see sources.")

    # --- Initialization and State Management ---
    if "collection_ready" not in st.session_state:
        st.session_state.collection_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = "knowledge_base"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "gemma3:12b" # Default model

    # Load essential resources; stop if any fail
    text_embedding_model = load_embedding_model()
    client = get_chroma_client()
    # LLM loading moved to where it's needed, based on selected model

    if not all([text_embedding_model, client]):
        st.error("Core components (Embedding Model or DB Client) failed to load. Cannot proceed.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection
        st.session_state.ollama_model = st.text_input(
            "Ollama Model Name",
            value=st.session_state.ollama_model,
            key="ollama_model_input",
            help="Ensure this Ollama model is pulled and available."
        )

        # Collection Management
        st.session_state.collection_name = st.text_input(
            "Vector Store Name",
            value=st.session_state.collection_name,
            key="collection_name_input",
            help="Documents will be loaded into this store (existing data overwritten on process)."
            # Simple reset on change for now
            # on_change=lambda: st.session_state.update(collection_ready=False, messages=[])
        )
        # Add button to explicitly clear/delete collection? Maybe later.

        st.divider()
        st.header("📚 Data Source")
        option = st.radio(
            "Choose source type:",
            ("Upload PDFs", "Search arXiv"),
            key="source_option",
            # Clear collection ready flag and messages when source type changes
            on_change=lambda: st.session_state.update(collection_ready=False, messages=[])
        )

        if option == "Upload PDFs":
            st.subheader("Upload PDF Files")
            uploaded_files = st.file_uploader(
                "Select one or more PDF files",
                accept_multiple_files=True,
                type=["pdf"],
                key="pdf_uploader"
            )
            if st.button("Process Uploaded PDFs", key="process_pdfs_button", disabled=not uploaded_files):
                with st.spinner("Extracting text from PDFs..."):
                    docs_with_metadata = extract_text_from_uploaded_pdfs(uploaded_files)
                if docs_with_metadata:
                    with st.spinner(f"Processing text into vector store '{st.session_state.collection_name}'..."):
                        collection = process_text_and_store(
                            client, text_embedding_model, docs_with_metadata, st.session_state.collection_name
                        )
                    if collection:
                        st.session_state.collection_ready = True
                        st.session_state.messages = [] # Reset chat
                        st.success(f"PDF content processed into '{st.session_state.collection_name}'!")
                    else:
                        st.error("Failed to process PDF content into the vector store.")
                        st.session_state.collection_ready = False
                else:
                    st.warning("No text could be extracted from the uploaded PDFs.")
                    st.session_state.collection_ready = False

        elif option == "Search arXiv":
            st.subheader("Search arXiv (Downloads PDF)")
            arxiv_query = st.text_input("Enter search query (e.g., 'attention is all you need'):", key="arxiv_search_query")
            max_results = st.number_input("Max papers to fetch", min_value=1, max_value=10, value=1, key="arxiv_max_results")

            if st.button("Search and Process ArXiv", key="process_arxiv_button", disabled=not arxiv_query):
                docs_with_metadata = []
                with st.spinner(f"Searching arXiv for '{arxiv_query}'..."):
                    try:
                        search = arxiv.Search(
                            query=arxiv_query,
                            max_results=max_results,
                            sort_by=arxiv.SortCriterion.Relevance
                        )
                        results = list(search.results()) # Get results as a list
                        st.session_state["arxiv_search_results_metadata"] = results # Store for potential display
                        logging.info(f"Found {len(results)} results on arXiv.")
                        if not results:
                            st.warning("No papers found matching your query on arXiv.")

                    except Exception as e:
                        logging.error(f"Error searching arXiv: {e}")
                        st.error(f"An error occurred during arXiv search: {e}")
                        results = []

                # Download and process PDFs found
                if results:
                    st.info(f"Found {len(results)} paper(s). Attempting to download and process PDFs...")
                    progress_bar = st.progress(0)
                    for i, result in enumerate(results):
                        st.write(f"Processing: {result.title}")
                        try:
                            pdf_filename = f"{result.entry_id.split('/')[-1].replace('.', '_')}.pdf" # Create filename
                            with st.spinner(f"Downloading PDF for '{result.title}'..."):
                                result.download_pdf(filename=pdf_filename) # Download to file system
                            logging.info(f"Downloaded PDF: {pdf_filename}")

                            # Read the downloaded PDF bytes
                            with open(pdf_filename, "rb") as f:
                                pdf_bytes = f.read()

                            # Extract text
                            source_id = result.pdf_url # Use URL as identifier
                            text, metadata = extract_text_from_pdf_bytes(pdf_bytes, source_id)
                            if text:
                                # Enhance metadata with arXiv info
                                metadata['title'] = result.title
                                metadata['authors'] = ', '.join(author.name for author in result.authors)
                                metadata['published'] = result.published.strftime('%Y-%m-%d')
                                metadata['summary'] = result.summary
                                metadata['source_type'] = 'arxiv'
                                docs_with_metadata.append((text, metadata))

                            # Clean up downloaded file
                            os.remove(pdf_filename)
                            logging.info(f"Removed temporary PDF: {pdf_filename}")

                        except Exception as e:
                            logging.error(f"Failed to download or process arXiv PDF {result.entry_id}: {e}")
                            st.warning(f"Could not process paper: {result.title} ({result.entry_id}). Skipping.")
                        progress_bar.progress((i + 1) / len(results))

                # Process all collected text into ChromaDB
                if docs_with_metadata:
                    with st.spinner(f"Processing text into vector store '{st.session_state.collection_name}'..."):
                         collection = process_text_and_store(
                            client, text_embedding_model, docs_with_metadata, st.session_state.collection_name
                        )
                    if collection:
                        st.session_state.collection_ready = True
                        st.session_state.messages = [] # Reset chat
                        st.success(f"arXiv content processed into '{st.session_state.collection_name}'!")
                    else:
                        st.error("Failed to process arXiv content into the vector store.")
                        st.session_state.collection_ready = False
                elif results: # Found results but none processed
                    st.error("Found papers on arXiv, but failed to download or process any PDFs.")
                    st.session_state.collection_ready = False

        st.divider()
        st.header("💬 Chat Controls")
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # --- Main Chat Area ---
    st.header("💬 Ask Questions")

    if not st.session_state.collection_ready:
        st.info("Please load documents using the sidebar first.")
    else:
        st.success(f"Ready to answer questions based on the content in '{st.session_state.collection_name}'.")

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Display context used for assistant messages, if available
                if message["role"] == "assistant" and "context_docs" in message:
                    with st.expander("Context Used"):
                        for doc, meta in zip(message["context_docs"], message["context_metas"]):
                            st.caption(f"Source: {meta.get('source', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')} | Title: {meta.get('title', 'N/A')[:50]}...")
                            st.markdown(f"```\n{doc[:300]}...\n```") # Show snippet

        # Chat input
        if prompt := st.chat_input("Ask a question...", disabled=not st.session_state.collection_ready):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Load LLM (cached based on selected model)
            llm = load_llm(st.session_state.ollama_model)
            if not llm:
                st.error("LLM failed to load. Cannot generate response.")
                st.session_state.messages.pop() # Remove user message
                st.stop()

            # Retrieve context
            context_docs, context_metas = None, None
            with st.spinner("Searching for relevant context..."):
                try:
                    current_collection = client.get_collection(name=st.session_state.collection_name)
                    context_docs, context_metas = semantic_search(current_collection, text_embedding_model, prompt, top_k=5)
                except Exception as e:
                    logging.error(f"Failed to get collection or search: {e}")
                    st.error(f"Error accessing vector store. Please try reprocessing the source.")
                    st.session_state.collection_ready = False
                    st.session_state.messages.pop()
                    st.rerun()

            # Build context string with metadata for the prompt
            context_str = ""
            if context_docs and context_metas:
                context_parts = []
                for doc, meta in zip(context_docs, context_metas):
                    source_info = f"Source: {meta.get('source', 'N/A')}, Chunk: {meta.get('chunk_id', 'N/A')}"
                    if 'title' in meta and meta['title'] != 'N/A':
                         source_info += f", Title: {meta['title']}"
                    context_parts.append(f"--- Context Chunk [{source_info}] ---\n{doc}\n--- End Chunk ---")
                context_str = "\n\n".join(context_parts)
            else:
                context_str = "No relevant context found in the documents."
                st.warning("Could not find relevant information to answer the query based on loaded documents.")

            # Generate response
            with st.spinner("Generating response..."):
                # Build history string
                history_str = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[:-1]]
                )

                # Enhanced Prompt Template
                template = """You are an AI assistant answering questions based *only* on the provided context chunks.
                Your goal is to provide accurate, concise answers derived solely from the text snippets given below.
                If the context does not contain the answer, state that clearly. Do not make up information.
                When possible, indicate the source of the information by referencing the 'Source' or 'Title' from the metadata provided within the chunk markers (e.g., "[Source: document.pdf, Chunk: 12]").

                Current conversation history:
                {history}

                Context from documents:
                {context}

                Human: {input}
                AI:"""
                prompt_template = PromptTemplate(input_variables=["history", "input", "context"], template=template)
                formatted_prompt = prompt_template.format(history=history_str, input=prompt, context=context_str)

                try:
                    response = llm.invoke(formatted_prompt)

                    # Store response and context used
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "context_docs": context_docs, # Store for display
                        "context_metas": context_metas  # Store for display
                    }
                    st.session_state.messages.append(assistant_message)

                    # Display AI response immediately (will be redisplayed with context on rerun)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        # Immediately show context for this response
                        with st.expander("Context Used for this Response"):
                             if context_docs and context_metas:
                                for doc, meta in zip(context_docs, context_metas):
                                    st.caption(f"Source: {meta.get('source', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')} | Title: {meta.get('title', 'N/A')[:50]}...")
                                    st.markdown(f"```\n{doc[:300]}...\n```")
                             else:
                                 st.write("No specific context chunks were retrieved for this response.")


                except Exception as e:
                    logging.error(f"Error generating response from LLM: {e}")
                    st.error(f"An error occurred while generating the response: {e}")
                    st.session_state.messages.pop() # Remove user msg if AI failed

            # Force a rerun to update the display including the context expander for previous messages
            # st.rerun() # Can sometimes cause usability issues, let's see if it works without it first.

    # Optionally display raw arXiv search metadata
    # if "arxiv_search_results_metadata" in st.session_state and option == "Search arXiv":
    #     with st.expander("Raw arXiv Search Metadata"):
    #          st.json([res.__dict__ for res in st.session_state.arxiv_search_results_metadata]) # Display fetched metadata


if __name__ == "__main__":
    # Install necessary libraries if you haven't:
    # pip install streamlit langchain langchain-community langchain-core pypdf2 sentence-transformers chromadb-client ollama requests arxiv urllib3
    main()