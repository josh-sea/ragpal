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
from qdrant_client import QdrantClient
from llama_index.core import PromptTemplate
from mail_manager import *
import tempfile
from typing import Any, Dict, Optional, Union, List, Optional


# Load environment variables
load_dotenv()

class LLMSource(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"

class RAGTool:
    """
    A tool for Retrieval-Augmented Generation (RAG) which initializes different language models, 
    sets up query engines, and ingests documents to be searchable.
    """
    def __init__(self, directory: str, model_name: str = "BAAI/bge-small-en-v1.5", llm_source: str = "local") -> None:
        """
        Initializes the RAGTool with the specified directory, model, and language model source.

        :param directory: The directory where the documents are stored.
        :param model_name: The name of the model to use for embeddings.
        :param llm_source: The source of the language model, can be 'openai', 'anthropic', 'mistral', or 'local'.
        """
        self._directory = directory
        self._model_name = model_name
        self._client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self._documents = []
        self._query_engines: Dict[str, RetrieverQueryEngine] = {}
        self._llm_source = llm_source
        self._llm = self.initialize_llm()
        self._embed_model = self.initialize_embed_model()
        self._document_type: Optional[str] = None
        
    def initialize_llm(self) -> Any:
        """
        Initializes the language model based on the _llm_source attribute.

        :return: An instance of the language model.
        """
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

    def file_mapping(self, type: str) -> str:
        """
        Maps a document type to its corresponding file extension.

        :param type: The type of document (e.g., 'javascript', 'python').
        :return: The file extension for the given document type.
        """
        language_map = {
            'javascript':'.js',
            'python':'.py',
            'pdf':'.pdf',
            'docx':'.docx',
            'csv':'.csv'
        }
        return language_map[type]

        
    def initialize_embed_model(self) -> HuggingFaceEmbedding:
        """
        Initializes the embedding model using HuggingFaceEmbedding.

        :return: An instance of HuggingFaceEmbedding.
        """
        embed_model = HuggingFaceEmbedding(model_name=self._model_name)
        Settings.embed_model = embed_model
        return embed_model
    
    @property
    def directory(self) -> str:
        """
        Gets the directory of the RAGTool.

        :return: The directory as a string.
        """
        return self._directory

    @directory.setter
    def directory(self, directory: str) -> None:
        """
        Sets the directory of the RAGTool.

        :param directory: The directory as a string.
        """
        self._directory = directory
            
    @property
    def document_type(self):
        """
        Gets the document_type of the RAGTool.

        :return: The document_type as a string.
        """
        return self._document_type

    @document_type.setter
    def document_type(self, document_type):
        """
        Sets the document_type of the RAGTool.

        :param document_type: The document_type as a string.
        """
        self._document_type = document_type

    @property
    def documents(self):
        """
        Gets the documents of the RAGTool.

        :return: The documents as a LlamaIndex Document.
        """
        return self._documents
    
    @documents.setter
    def documents(self, documents):
        """
        Sets the documents of the RAGTool.

        :param documents: The documents as a LlamaIndex Document.
        """
        self._documents = documents
        
        
    def create_temp_json_file(self, json_data: Any) -> str:
        """
        Creates a temporary JSON file with the provided JSON data.

        :param json_data: The JSON data to be written to the file.
        :return: The file path of the created temporary JSON file.
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Define the file path
        file_path = os.path.join(temp_dir, 'data.json')
        # Write the JSON data to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        return file_path
    
    def process_json_data_with_reader(self, json_data: Any) -> List:
        """
        Processes JSON data by reading it through a JSONReader, and optionally cleans up the created temporary files.

        :param json_data: The JSON data to be processed.
        :return: A list of document nodes obtained from processing the JSON data.
        """
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


    def _load_documents(self, document_type: str, directory: Optional[str] = None, crawl_depth: int = 0, after: Optional[datetime] = None) -> None:
        """
        Loads documents based on the specified document type and additional parameters for document sourcing.

        :param document_type: The type of documents to load ('web', 'email', or others).
        :param directory: Optional directory or URL from where to load the documents.
        :param crawl_depth: The depth for web crawling, applicable for web documents.
        :param after: The datetime for fetching emails after this time, applicable for emails.
        """
        if document_type == "web":
            if directory:
                url = directory
                self.directory = url
            loader = WholeSiteReader(prefix=url, max_depth=crawl_depth)
            documents = self.clean_documents(loader.load_data(base_url=url))
            self.documents = documents
        elif document_type == "email":
            self.document_type = "email"
            print(f"in _load_documents in app.py: {after}")
            new_emails = fetch_latest_emails(after)
            documents = self.clean_documents(self.process_json_data_with_reader(new_emails))
            self.documents = documents
        else:    
            if directory:
                self.directory = directory
            file_extension = self.file_mapping(document_type)
            reader = SimpleDirectoryReader(self.directory, recursive=True, required_exts=[file_extension])
            self.documents = self.clean_documents(reader.load_data())

    def clean_up_text(self, content: str) -> str:
        """
        Cleans up the text content by removing unnecessary characters and formatting.

        :param content: The original text content.
        :return: The cleaned-up text content.
        """
        # cleans up line breaks, etc. 
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)
        unwanted_patterns = ["\\n", "  —", "——————————", "—————————", "—————", r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7']
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)
        return content

    def clean_documents(self, documents: List) -> List:
        """
        Processes a list of documents to clean up the text content.

        :param documents: A list of document nodes.
        :return: A list of cleaned document nodes.
        """
        # process all document context to clean up line breaks and wasteful character patterns
        cleaned_docs = []
        for d in documents:
            cleaned_text = self.clean_up_text(d.text)
            d.text = cleaned_text
            cleaned_docs.append(d)
        
        return cleaned_docs

    @property
    def node_parser(self):
        """
        Gets the node parser of the RAGTool.

        :return: The node parser as a NodeParser.
        """
        return self._node_parser
        
    @node_parser.setter
    def node_parser(self, document_type):
        """
        Sets the node parser of the RAGTool as a CodeSplitter, JSONNodeParser, or SemanticSplitterNodeParser (inherits from NodeParser)

        :param document_type: The node parser as a string.
        """
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

    def setup_query_engine(self, document_type: str, vector_store: Optional[QdrantVectorStore] = None) -> None:
        """
        Sets up a query engine for a specific document type.

        :param document_type: The type of document the query engine will be used for.
        :param vector_store: An optional instance of QdrantVectorStore to be used with the query engine.
        """
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
    
    def run_pipeline(self, document_type: str, directory: Optional[str] = None, crawl_depth: int = 0, after: Optional[datetime] = None) -> None:
        """
        Runs the ingestion pipeline for a specific document type.

        :param document_type: The type of documents to process.
        :param directory: Optional directory containing documents to process.
        :param crawl_depth: The depth for web crawling (applicable for web documents).
        :param after: The datetime for fetching emails after this time (applicable for emails).
        """
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


def initialize_rag_tool(directory: str = "", llm_source: str = "openai") -> RAGTool:
    """
    Initializes and returns an instance of the RAGTool class.

    :param directory: The directory where the documents are stored.
    :param llm_source: The source of the language model, can be 'openai', 'anthropic', 'mistral', or 'local'.
    :return: An instance of RAGTool.
    """
    # Initialize and configure the RAGTool instance
    rag_tool = RAGTool(directory=directory, llm_source=llm_source)
    # You can run any initial setup here if necessary
    return rag_tool


if __name__ == '__main__':
    # For example, a demonstration of the tool's functionality
    rag_tool_demo = initialize_rag_tool()
