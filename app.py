import re
import os
from enum import Enum
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, Settings, IngestionPipeline)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import CodeSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
import qdrant_client

# Load environment variables
load_dotenv()

class LLMSource(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"

class RAGTool:
    def __init__(self, directory, model_name="BAAI/bge-small-en-v1.5", llm_source=LLMSource.LOCAL):
        self.directory = directory
        self.model_name = model_name
        self.embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        self.client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self.documents = []
        self.query_engines = {}
        self.llm_source = llm_source
        self.llm = self.initialize_llm()
        
    def initialize_llm(self):
        if self.llm_source == LLMSource.OPENAI:
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.llm_source == LLMSource.ANTHROPIC:
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-opus-20240229")
        elif self.llm_source == LLMSource.MISTRAL:
            # Assuming MistralAI is a class similar to OpenAI and Anthropic
            return MistralAI(api_key=os.getenv("MISTRAL_API_KEY"))
        elif self.llm_source == LLMSource.LOCAL:
            return OpenAI(model="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")
        else:
            raise ValueError(f"Unsupported LLM source: {self.llm_source}")

    def load_documents(self):
        reader = SimpleDirectoryReader(self.directory, recursive=True)
        self.documents = reader.load_data()

    def clean_up_text(content: str) -> str:
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)
        unwanted_patterns = ["\\n", "  —", "——————————", "—————————", "—————", r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7']
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)
        return content

    def setup_pipelines(self):
        cleaned_docs = self.clean_documents()
        self.setup_javascript_pipeline(cleaned_docs)

    def clean_documents(self):
        cleaned_docs = []
        for d in self.documents:
            file_type = os.path.splitext(d.path)[1]
            if file_type == '.js':
                cleaned_text = self.clean_up_text(d.text)
                d.text = cleaned_text
                cleaned_docs.append(d)
        return cleaned_docs

    def setup_javascript_pipeline(self, documents):
        collection_name = "javascript_code"
        vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)

        # Specific settings for JavaScript, such as node parsing, could be initialized here
        splitter = CodeSplitter(
            language="javascript",
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=1500,
        )
        Settings.node_parser = splitter

        pipeline = IngestionPipeline(vector_store=vector_store)
        pipeline.run(documents=documents)

        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        self.query_engines['javascript'] = query_engine

    def query(self, query_text, file_type):
        if file_type not in self.query_engines:
            raise ValueError(f"No query engine set up for file type: {file_type}")
        response = self.query_engines[file_type].query(query_text)
        return response

rag_tool = RAGTool(directory="./docs", llm_source=LLMSource.ANTHROPIC)
rag_tool.load_documents()
rag_tool.setup_pipelines()
response = rag_tool.query("Your query here", 'javascript')  # Assuming the query is intended for JavaScript documents
print(response)
