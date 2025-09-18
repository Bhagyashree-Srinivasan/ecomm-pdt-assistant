import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

from utils.config_loader import load_config
from utils.model_loader import ModelLoader


class Retriever:

    def __init__(self):
        """
        Initialize the Retriever class by loading configurations.
        """
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vector_store = None
        self.retriever = None

    def _load_env_variables(self):
        """
        Load environment variables from a .env file. 
        """
        load_dotenv()

        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
         
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.gogle_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def load_retriever(self):
        """
        Load the vector store and initialize the retriever.
        """
        if not self.vector_store:
            collection_name = self.config["astra_db"]["collection_name"]
            self.vector_store = AstraDBVectorStore(
                embedding = self.model_loader.load_embeddings(),
                collection_name = collection_name,
                api_endpoint = self.db_api_endpoint,
                token = self.db_application_token,
                namespace = self.db_keyspace
            )
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retreiver" in self.config else 3
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            print("Retriever loaded successfully.")
            return retriever

    def call_retriever(self, query):
        """
        Invoke the retriever to get relevant documents.
        """
        retriever = self.load_retriever()
        output = retriever.invoke(query)
        return output


if __name__ == "__main__":
    retriever_obj = Retriever()
    user_query = "Can you suggest good budget phones?"
    results = retriever_obj.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Results {idx}: {doc.page_content}\n Metadata: {doc.metadata}\n") 