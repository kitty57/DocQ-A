import streamlit as st
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
import os
import textwrap  # Import the textwrap module

# Ensure event loop is properly initialized
asyncio.set_event_loop(asyncio.new_event_loop())

# Function to perform question answering
def perform_question_answering(uploaded_files, question):
    # Check if documents are uploaded
    if uploaded_files:
        directory = "uploaded_documents"
        os.makedirs(directory, exist_ok=True)
        for i, uploaded_file in enumerate(uploaded_files):
            with open(os.path.join(directory, f"document_{i}.pdf"), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Initialize LLM and embedding models
        llm = GradientBaseModelLLM(
            base_model_slug="llama2-7b-chat",
            max_tokens=400,
        )
        embed_model = GradientEmbedding(
            gradient_access_token=st.secrets["GRADIENT_ACCESS_TOKEN"],
            gradient_workspace_id=st.secrets["GRADIENT_WORKSPACE_ID"],
            gradient_model_slug="bge-large",
        )

        # Load documents into VectorStoreIndex
        documents_reader = SimpleDirectoryReader(directory).load_data()
        vector_store_index = VectorStoreIndex.from_documents(documents_reader, llm=llm, embed_model=embed_model)
        query_engine = vector_store_index.as_query_engine()

        # Perform question answering
        response = query_engine.query(question)

        return response

def main():
    st.set_page_config(page_title="Document Q&A Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded", menu_items={"Get Help": None, "Report a Bug": None})
    
    st.title("Document Q&A Chatbot")
    
    # Customizing the background
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://example.com/background.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Sidebar for uploading PDF documents
    st.sidebar.title("Upload PDF Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    # Chat interface
    st.title("Chat Interface")
    question = st.text_input("You: ", "")

    # Answer button
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            # Perform question answering
            response = perform_question_answering(uploaded_files, question)
            if response:
                # Wrap the response text to ensure it displays properly
                wrapped_text = textwrap.fill(response.response, width=70)
                st.text("Bot: " + wrapped_text)  # Display the wrapped text
                question = ""  # Clear the input box after answering
            else:
                st.text("Bot: Sorry, I couldn't find an answer.")

if __name__ == "__main__":
    main()
