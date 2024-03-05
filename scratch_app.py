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
from llama_index.llms.mistralai import MistralAI
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
        self.client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self.documents = []
        self.query_engines = {}
        self.llm_source = llm_source
        self.llm = self.initialize_llm()
        self.embed_model = self.initialize_embed_model()

    def initialize_llm(self):
        if self.llm_source == LLMSource.OPENAI:
            llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo-preview")
        elif self.llm_source == LLMSource.ANTHROPIC:
            llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-opus-20240229")
        elif self.llm_source == LLMSource.MISTRAL:
            # Assuming MistralAI is a class similar to OpenAI and Anthropic
            llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"))
        elif self.llm_source == LLMSource.LOCAL:
            llm = OpenAI(model="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")
        else:
            raise ValueError(f"Unsupported LLM source: {self.llm_source}")

        Settings.llm = llm
        return llm
    
    def initialize_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model
        return embed_model
    
    def clean_up_text(self, content: str) -> str:
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)
        unwanted_patterns = ["\\n", "  —", "——————————", "—————————", "—————", r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7']
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)
        return content

    def clean_documents(self):
        cleaned_docs = []
        for d in self.documents:
            cleaned_text = self.clean_up_text(d.text)
            d.text = cleaned_text
            cleaned_docs.append(d)
        return cleaned_docs

    def load_and_process_documents(self, file_extension, language):
        # Load documents with the specified file extension
        reader = SimpleDirectoryReader(self.directory, recursive=True, required_exts=[file_extension])
        documents = reader.load_data()
        # Clean documents
        cleaned_docs = []
        for d in documents:
            cleaned_text = self.clean_up_text(d.text)
            d.text = cleaned_text
            cleaned_docs.append(d)

        # Setup pipeline for the specific language
        self.setup_pipeline(cleaned_docs, language)

    def setup_pipeline(self, documents, language):
        collection_name = f"{language}_code"
        vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)

        # Language-specific settings, such as node parsing
        splitter = CodeSplitter(
            language=language,
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=1500,
        )
        Settings.node_parser = splitter

        # Check if the pipeline needs to be created or just retrieved
        if language not in self.query_engines:
            pipeline = IngestionPipeline(vector_store=vector_store)
            pipeline.run(documents=documents)

            vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
            response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )

            self.query_engines[language] = query_engine
        else:
            # Query engine already exists, no need to create a new pipeline
            print(f"Query engine for {language} already initialized.")

    def query(self, query_text, file_type):
        # Map file extensions to languages
        language_map = {
            '.js': 'javascript',
            '.py': 'python'
        }

        if file_type not in language_map:
            raise ValueError(f"No query engine set up for file type: {file_type}")

        language = language_map[file_type]

        if language not in self.query_engines:
            raise ValueError(f"No query engine initialized for language: {language}")

        response = self.query_engines[language].query(query_text)
        return response

# Example usage:
rag_tool = RAGTool(directory="./docs", llm_source=LLMSource.MISTRAL)
rag_tool.load_and_process_documents(".py", "python")
# rag_tool.load_and_process_documents(".js", "javascript")

# Querying the system
response = rag_tool.query("How does the privateGPT handle context retrieval?", ".py")
print(response)

