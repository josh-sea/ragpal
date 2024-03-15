import re
import os
import requests
from enum import Enum
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, Settings)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import CodeSplitter, SemanticSplitterNodeParser, JSONNodeParser
from llama_index.readers.web import BeautifulSoupWebReader, WholeSiteReader
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import MultiStepQueryEngine
import qdrant_client
from llama_index.core import PromptTemplate
from mail_manager import *
import tempfile
from typing import Any


# Load environment variables
load_dotenv()

class LLMSource(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"

class RAGTool:
    def __init__(self, directory, model_name="BAAI/bge-small-en-v1.5", llm_source="local"):
        self._directory = directory
        self._model_name = model_name
        self._client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self._documents = []
        self._query_engines = {}
        self._llm_source = llm_source
        self._llm = self.initialize_llm()
        self._embed_model = self.initialize_embed_model()
        self._document_type = None
        
    def initialize_llm(self):
        if self._llm_source == "openai":
            llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo-preview")
        elif self._llm_source == "anthropic":
            llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-opus-20240229")
        elif self._llm_source == "mistral":
            # Assuming MistralAI is a class similar to OpenAI and Anthropic
            llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"))
        elif self._llm_source == "local":
            llm = OpenAI(model="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")
        else:
            raise ValueError(f"Unsupported LLM source: {self._llm_source}")

        Settings.llm = llm
        return llm

    def file_mapping(self, type):
        language_map = {
            'javascript':'.js',
            'python':'.py',
            'pdf':'.pdf',
            'docx':'.docx',
            'csv':'.csv'
        }
        return language_map[type]

        
    def initialize_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self._model_name)
        Settings.embed_model = embed_model
        return embed_model
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, directory):
        self._directory = directory
            
    @property
    def document_type(self):
        return self._document_type

    @document_type.setter
    def document_type(self, document_type):
        self._document_type = document_type

    @property
    def documents(self):
        return self._documents
    
    @documents.setter
    def documents(self, documents):
        self._documents = documents
        
        
    def create_temp_json_file(self, json_data: Any) -> str:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Define the file path
        file_path = os.path.join(temp_dir, 'data.json')
        # Write the JSON data to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        return file_path
    
    def process_json_data_with_reader(self, json_data: Any):
        # First, create a temporary JSON file
        json_file_path = self.create_temp_json_file(json_data)
        # Initialize your JSONReader (with desired configuration)
        json_reader = JSONReader(levels_back=1, collapse_length=100, ensure_ascii=False)
        # Process the data
        documents = json_reader.load_data(json_file_path)
        # Optionally, cleanup the temporary file and directory if desired
        os.remove(json_file_path)
        os.rmdir(os.path.dirname(json_file_path))
        
        return documents


    def _load_documents(self, document_type: str, directory: str=None, crawl_depth: int=0, after=None):
        if document_type == "web":
            if directory:
                url = directory
                self.directory = url
            # loader = BeautifulSoupWebReader()
            # documents = loader.load_data(urls=[url])
            loader = WholeSiteReader(prefix=url, max_depth=crawl_depth)
            documents = self.clean_documents(loader.load_data(base_url=url))
            # Initialize the scraper with a prefix URL and maximum depth
            self.documents = documents
        elif document_type == "email":
            self.document_type = "email"
            print(f"in _load_documents in app.py: {after}")
            new_emails = fetch_latest_emails(after)  # Your email fetching function 
            documents = self.clean_documents(self.process_json_data_with_reader(new_emails))
            # Set up for email processing
            self.documents = documents
        else:    
            if directory:
                self.directory = directory
            file_extension = self.file_mapping(document_type)
            reader = SimpleDirectoryReader(self.directory, recursive=True, required_exts=[file_extension])
            self.documents = self.clean_documents(reader.load_data())

    def clean_up_text(self, content: str) -> str:
        # cleans up line breaks, etc. 
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)
        unwanted_patterns = ["\\n", "  —", "——————————", "—————————", "—————", r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7']
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)
        return content

    def clean_documents(self, documents) -> list:
        # process all document context to clean up line breaks and wasteful character patterns
        cleaned_docs = []
        for d in documents:
            cleaned_text = self.clean_up_text(d.text)
            d.text = cleaned_text
            cleaned_docs.append(d)
        
        return cleaned_docs

    @property
    def node_parser(self):
        return self._node_parser
        
    @node_parser.setter
    def node_parser(self, document_type):
        if document_type == "javascript":
            node_parser = CodeSplitter(
                language="javascript",
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
            self._node_parser = node_parser
            Settings.node_parser = node_parser 
        
        if document_type == "email":
            # JSONNodeParser does not require specific initialization for emails
            node_parser = JSONNodeParser()
            self._node_parser = node_parser
            Settings.node_parser = node_parser
            
        if document_type == "python":
            node_parser = CodeSplitter(
                language="python",
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
            self._node_parser = node_parser
            Settings.node_parser = node_parser 
            
        if document_type == "text":
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
            )
            self._node_parser = node_parser
            Settings.node_parser = node_parser
        
        if document_type == "web":
            node_parser = CodeSplitter(
                language="html",
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
            self._node_parser = node_parser
            Settings.node_parser = node_parser
    
    def get_or_create_query_engine(self, document_type, vector_store=None):
        if document_type not in self._query_engines:
            self.setup_query_engine(document_type, vector_store)
        return self._query_engines[document_type]

    def setup_query_engine(self, document_type, vector_store=None):
        if document_type not in self._query_engines:
            if vector_store == None:
                vector_store = QdrantVectorStore(client=self._client, collection_name=document_type)
            
            vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=5
            )
            response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
            query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
            self._query_engines[document_type] = query_engine
    
    def run_pipeline(self, document_type: str, directory: str=None, crawl_depth: int=0, after=None):
        print(f"in run pipeline in app.py: {after}")
        self._load_documents(document_type, directory, crawl_depth, after)
        self.node_parser = self.document_type
        # if document_type == "email":
            # parser = self.node_parser
            # self.documents = parser.get_nodes_from_documents(self.documents)
        # documents_to_ingest = self.clean_documents() added cleaning to loading so stored documents are cleaned already
        vector_store = QdrantVectorStore(client=self._client, collection_name=document_type)
        pipeline = IngestionPipeline(vector_store=vector_store)
        pipeline.run(documents=self.documents, num_workers=2)
        
        self.get_or_create_query_engine(document_type, vector_store)
        
    def query(self, query_text, document_type):
        query_engine = self.get_or_create_query_engine(document_type)
        response = query_engine.query(query_text)
        return response

def initialize_rag_tool(directory="", llm_source="openai"):
    # Initialize and configure the RAGTool instance
    rag_tool = RAGTool(directory=directory, llm_source=llm_source)
    # You can run any initial setup here if necessary
    return rag_tool


if __name__ == '__main__':
    # For example, a demonstration of the tool's functionality
    rag_tool_demo = initialize_rag_tool()
