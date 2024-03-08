import streamlit as st
from app import initialize_rag_tool
import os
from PIL import Image
import tempfile

# Initialize your RAG tool outside the main app function to maintain state across sessions
# rag_tool = RAGTool(directory="./docs", llm_source="anthropic")  # Adjust llm_source as needed

def run_pipeline_wrapper(rag_tool, document_type, directory=None, upload=None):
    # Handle document upload or directory specification
    if upload:
        print("getting file tmp path")
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in upload:
                bytes_data = uploaded_file.getvalue()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
            # Update the directory to the temp directory with uploaded files
            print("running pipeline")
            rag_tool.run_pipeline(document_type, temp_dir)
    elif directory:
        rag_tool.run_pipeline(document_type, directory)
    else:
        st.error("Please upload documents or specify a directory.")
        
        
def estimate_height(text, line_height=20, padding=20, min_height=75):
    lines = text.count('\n') + 1  # Count how many lines are in the text
    estimated_height = lines * line_height + padding  # Calculate height based on lines and line height
    return max(estimated_height, min_height)  # Return the larger of estimated height or min_height

def main():
    rag_tool = initialize_rag_tool(directory="./docs", llm_source="anthropic")
    st.title("RAG Tool Chat Interface")

    # Sidebar for pipeline controls and settings
    with st.sidebar:
        st.header("Settings")
        run_pipeline = st.checkbox("Run Pipeline", help="Toggle this to upload documents. Only run when new documents are added.")
        document_type = st.selectbox("Document Type", options=["web", "javascript", "python", "pdf", "docx", "csv"], index=0)
        directory = st.text_input("Document Directory (Optional)", placeholder="/path/to/documents")
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'js', 'py'])

    # Chat interface
    conversation_history = st.session_state.get("conversation_history", [])
    user_input = st.text_input("Your Question:", placeholder="Type your question here...")

    if st.button("Send"):
        if user_input:
            # Display user's question
            conversation_history.append(f"You: {user_input}")
            if run_pipeline:
                run_pipeline_wrapper(rag_tool, document_type, directory, uploaded_files)

            # Generate and display the response
            response = rag_tool.query(user_input, document_type)
            conversation_history.append(f"RAG: {response}")

            # Update conversation history in state
            st.session_state.conversation_history = conversation_history

        # Display conversation history
        for chat in conversation_history:
            height = estimate_height(chat)
            st.text_area("", chat, height=height, key=chat[:10])


if __name__ == "__main__":
    main()