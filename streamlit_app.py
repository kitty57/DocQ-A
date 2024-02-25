import streamlit as st
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
import os

# Retrieve Gradient Access Token and Workspace ID from environment variables
gradient_access_token = os.getenv('GRADIENT_ACCESS_TOKEN')
gradient_workspace_id = os.getenv('GRADIENT_WORKSPACE_ID')

def main():
    st.set_page_config(page_title="Document Q&A App", page_icon="📚")
    if gradient_access_token and gradient_workspace_id:
        llm = GradientBaseModelLLM(
            base_model_slug="llama2-7b-chat",
            max_tokens=400,
        )
        embed_model = GradientEmbedding(
            gradient_access_token=gradient_access_token,
            gradient_workspace_id=gradient_workspace_id,
            gradient_model_slug="bge-large",
        )

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=256,
        )
        set_global_service_context(service_context)

        st.title("Document Q&A App")
        st.sidebar.title("Upload PDF Documents")
        uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

        if uploaded_files:
            directory = "uploaded_documents"
            os.makedirs(directory, exist_ok=True)
            for i, uploaded_file in enumerate(uploaded_files):
                with open(os.path.join(directory, f"document_{i}.pdf"), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            documents_reader = SimpleDirectoryReader(directory)
            vector_store_index = VectorStoreIndex.from_documents(documents_reader, service_context=service_context)
            query_engine = vector_store_index.as_query_engine()

            st.success("Documents processed successfully!")

            question = st.text_input("Ask your question:", "")

            if st.button("Get Answer"):
                with st.spinner("Searching..."):
                    response = query_engine.query(question)
                    if response:
                        st.write("Answer:")
                        st.write(response[0].response)
                    else:
                        st.write("Sorry, no answer found.")

if __name__ == "__main__":
    main()
