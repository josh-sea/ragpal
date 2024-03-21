import time
import uuid
import json
import streamlit as st
from app import initialize_rag_tool
import os
from PIL import Image
import tempfile

# Initialize your RAG tool outside the main app function to maintain state across sessions
# rag_tool = RAGTool(directory="./docs", llm_source="anthropic")  # Adjust llm_source as needed

def run_pipeline_wrapper(rag_tool, document_type, directory=None, upload=None, crawl_depth=None):
    # Handle document upload or directory specification
    if document_type == "email":
        after = load_last_query_timestamp()
        rag_tool.run_pipeline(document_type, directory, crawl_depth, after)
    elif upload:
        if not crawl_depth:
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
        else:
            rag_tool.run_pipeline(document_type, temp_dir, crawl_depth)
    elif directory:
        rag_tool.run_pipeline(document_type, directory)
    else:
        st.error("Please upload documents or specify a directory.")
        
def save_last_query_timestamp(timestamp=None):    
    # Save to Streamlit's session state
    st.session_state['last_query_timestamp'] = timestamp

def load_last_query_timestamp():
    # Try loading from Streamlit's session state
    if 'last_query_timestamp' in st.session_state:
        return st.session_state['last_query_timestamp']
    # Load from a local JSON file
    try:
        with open('last_query_timestamp.json', 'r') as file:
            data = json.load(file)
            save_last_query_timestamp(data.get('last_query_timestamp'))
            return st.session_state['last_query_timestamp']
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def estimate_height(text, line_height=20, padding=20, min_height=75):
    lines = text.count('\n') + 1  # Count how many lines are in the text
    estimated_height = lines * line_height + padding  # Calculate height based on lines and line height
    return max(estimated_height, min_height)  # Return the larger of estimated height or min_height

def main():
    rag_tool = initialize_rag_tool(directory="./docs", llm_source="anthropic")
    st.set_page_config(layout="wide", initial_sidebar_state="auto", page_icon="🍕")
    st.title("RAG Tool Chat Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])


    # Sidebar for pipeline controls and settings
    with st.sidebar:
        st.header("Settings")
        run_pipeline = st.checkbox("Run Pipeline", help="Toggle this to upload documents. Only run when new documents are added.")
        document_type = st.selectbox("Document Type", options=["web", "email", "javascript", "python", "pdf", "docx", "csv"], index=0)
        if document_type == "web":
            crawl_depth = st.number_input("Web Crawl Depth (Default 0 - just the url provided)", placeholder=0, min_value=0, step=1)
            uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'js', 'py'], disabled=True)
            directory = st.text_input("URL for web", placeholder="https://www.google.com")
            upload_button = st.button("Crawl")
        elif document_type == "email":
            uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'js', 'py'], disabled=True)
            upload_button = st.button("Fetch latest emails")
        else:
            # directory = st.text_input("Document Directory (Optional) - URL for web", placeholder="path/to/document")
            uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'js', 'py'], disabled=False)
            upload_button = st.button("Upload")

    # Chat interface
    conversation_history = st.session_state.get("conversation_history", [])

    if upload_button:
       if run_pipeline:
           with st.spinner('Running pipeline'):
               if document_type != "web":
                   directory=None
                   run_pipeline_wrapper(rag_tool, document_type, directory, uploaded_files) 
               elif document_type == "email":
                   directory=None
                   uploaded_files=None
                   run_pipeline_wrapper(rag_tool, document_type, directory, uploaded_files) 
               else:
                   run_pipeline_wrapper(rag_tool, document_type, directory, uploaded_files, crawl_depth) 

    if user_input := st.chat_input("Type your question here..."):
        # Display user's question
        with st.chat_message("user", avatar="💪"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "💪"})
    
    
        with st.chat_message("assistant", avatar="🦾"):
            # message_placeholder = st.empty()
            # full_response = ""
            with st.spinner('working on it...'):
                conversation = ""
                for message in st.session_state.messages:
                    role = message["role"]
                    content = message["content"]
                    conversation += f"role: {role}, content: {content}\n"  # Appending '\n' for a new line for readability
                    
                assistant_response = rag_tool.query(conversation, document_type)
            
            st.markdown(assistant_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response, "avatar": "🦾"})
                # print(str(assistant_response).split())
            # for chunk in str(assistant_response).split():
            #     full_response += chunk + " "
            #     time.sleep(0.05)
            #         # Add a blinking cursor to simulate typing
            #     message_placeholder.markdown(full_response + "▌")
            #     message_placeholder.markdown(full_response)
            
        

    # Display conversation history
    # for owner, chat, chat_id in conversation_history:
    #     height = estimate_height(chat)
    #     if owner == "user":
    #         message = st.chat_message("user", avatar="😎")
    #         message.write(chat)
    #     else: 
    #         message = st.chat_message("ai", avatar="🦾")
    #         message.write(chat)
    #         # st.text_area("", chat, height=height, key=chat_id)

if __name__ == "__main__":
    main()