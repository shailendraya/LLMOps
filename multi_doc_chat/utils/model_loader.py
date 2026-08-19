import os
import sys
import json

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_groq import ChatGroq

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class ApiKeyManager:
    REQUIRED_KEYS = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
    ]

    def __init__(self):
        self.api_keys = {}

        # ---------------------------------------------------------
        # Load API keys from JSON secret
        # ---------------------------------------------------------
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "apikeyliveclass is not a valid JSON object"
                    )

                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")

            except Exception as e:
                log.warning(
                    "Failed to parse API_KEYS as JSON",
                    error=str(e),
                )

        # ---------------------------------------------------------
        # Fallback to individual environment variables
        # ---------------------------------------------------------
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)

                if env_val:
                    self.api_keys[key] = env_val
                    log.info(
                        f"Loaded {key} from individual env var"
                    )

        # ---------------------------------------------------------
        # Validate required keys
        # ---------------------------------------------------------
        missing = [
            key
            for key in self.REQUIRED_KEYS
            if not self.api_keys.get(key)
        ]

        if missing:
            log.error(
                "Missing required API keys",
                missing_keys=missing,
            )
            raise DocumentPortalException(
                "Missing API keys",
                sys,
            )

        # Do not log complete API keys
        log.info(
            "API keys loaded",
            keys={
                key: value[:6] + "..."
                for key, value in self.api_keys.items()
            },
        )

    def get(self, key: str) -> str:
        value = self.api_keys.get(key)

        if not value:
            raise KeyError(
                f"API key for {key} is missing"
            )

        return value


class ModelLoader:
    """
    Loads embedding models and LLMs.

    IMPORTANT:
    -------------------------------------------------------------
    Document embedding:
        task_type = RETRIEVAL_DOCUMENT

    Query embedding:
        task_type = RETRIEVAL_QUERY

    These MUST remain separate for RAG retrieval.
    """

    def __init__(self):
        # ---------------------------------------------------------
        # Environment
        # ---------------------------------------------------------
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info(
                "Running in LOCAL mode: .env loaded"
            )
        else:
            log.info(
                "Running in PRODUCTION mode"
            )

        # ---------------------------------------------------------
        # API keys
        # ---------------------------------------------------------
        self.api_key_mgr = ApiKeyManager()

        # ---------------------------------------------------------
        # Config
        # ---------------------------------------------------------
        self.config = load_config()

        log.info(
            "YAML config loaded",
            config_keys=list(self.config.keys()),
        )

    # =============================================================
    # EMBEDDINGS
    # =============================================================

    def _get_embedding_model_name(self):
        """
        Read embedding model name from YAML config.
        """

        return self.config[
            "embedding_model"
        ]["model_name"]

    def load_document_embeddings(self):
        """
        Load embedding model for DOCUMENTS.

        Used during:
            PDF/TXT ingestion
            chunk embedding
            FAISS index creation

        task_type:
            RETRIEVAL_DOCUMENT
        """

        try:
            model_name = self._get_embedding_model_name()

            log.info(
                "Loading document embedding model",
                model=model_name,
                task_type="RETRIEVAL_DOCUMENT",
            )

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get(
                    "GOOGLE_API_KEY"
                ),
                task_type="RETRIEVAL_DOCUMENT",
            )

            log.info(
                "Embedding model loaded successfully",
                model=model_name,
                task_type="RETRIEVAL_DOCUMENT",
            )

            return embeddings

        except Exception as e:
            log.error(
                "Error loading document embedding model",
                error=str(e),
            )

            raise DocumentPortalException(
                "Failed to load document embedding model",
                sys,
            )

    def load_query_embeddings(self):
        """
        Load embedding model for USER QUERIES.

        Used during:
            similarity_search()
            retriever.invoke()
            FAISS query embedding

        task_type:
            RETRIEVAL_QUERY
        """

        try:
            model_name = self._get_embedding_model_name()

            log.info(
                "Loading query embedding model",
                model=model_name,
                task_type="RETRIEVAL_QUERY",
            )

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get(
                    "GOOGLE_API_KEY"
                ),
                task_type="RETRIEVAL_QUERY",
            )

            log.info(
                "Query embedding model loaded successfully",
                model=model_name,
                task_type="RETRIEVAL_QUERY",
            )

            return embeddings

        except Exception as e:
            log.error(
                "Error loading query embedding model",
                error=str(e),
            )

            raise DocumentPortalException(
                "Failed to load query embedding model",
                sys,
            )

    # -------------------------------------------------------------
    # Backward compatibility
    # -------------------------------------------------------------
    #
    # DO NOT use this method for new code.
    #
    # Existing ingestion code that calls:
    #
    #     load_embeddings()
    #
    # will continue to get DOCUMENT embeddings.
    #
    # Retrieval code MUST explicitly use:
    #
    #     load_query_embeddings()
    #
    # -------------------------------------------------------------

    def load_embeddings(self):
        """
        Backward-compatible alias.

        Returns DOCUMENT embeddings.

        New code should explicitly call:
            load_document_embeddings()
        or:
            load_query_embeddings()
        """

        return self.load_document_embeddings()

    # =============================================================
    # LLM
    # =============================================================

    def load_llm(self):
        """
        Load configured LLM.
        """

        llm_block = self.config["llm"]

        provider_key = os.getenv(
            "LLM_PROVIDER",
            "google",
        )

        if provider_key not in llm_block:
            log.error(
                "LLM provider not found in config",
                provider=provider_key,
            )

            raise ValueError(
                f"LLM provider '{provider_key}' "
                f"not found in config"
            )

        llm_config = llm_block[provider_key]

        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")

        temperature = llm_config.get(
            "temperature",
            0.2,
        )

        max_tokens = llm_config.get(
            "max_output_tokens",
            2048,
        )

        log.info(
            "Loading LLM",
            provider=provider,
            model=model_name,
        )

        # ---------------------------------------------------------
        # Google Gemini
        # ---------------------------------------------------------
        if provider == "google":

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get(
                    "GOOGLE_API_KEY"
                ),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        # ---------------------------------------------------------
        # Groq
        # ---------------------------------------------------------
        elif provider == "groq":

            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get(
                    "GROQ_API_KEY"
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        else:

            log.error(
                "Unsupported LLM provider",
                provider=provider,
            )

            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )


# =============================================================
# LOCAL TEST
# =============================================================

if __name__ == "__main__":

    loader = ModelLoader()

    # ---------------------------------------------------------
    # Test DOCUMENT embedding
    # ---------------------------------------------------------

    print("\n========================================")
    print("Testing DOCUMENT embedding")
    print("========================================")

    document_embeddings = (
        loader.load_document_embeddings()
    )

    document_vector = (
        document_embeddings.embed_query(
            "This is a document about transformers."
        )
    )

    print(
        f"Document embedding dimensions: "
        f"{len(document_vector)}"
    )

    # ---------------------------------------------------------
    # Test QUERY embedding
    # ---------------------------------------------------------

    print("\n========================================")
    print("Testing QUERY embedding")
    print("========================================")

    query_embeddings = (
        loader.load_query_embeddings()
    )

    query_vector = (
        query_embeddings.embed_query(
            "What is a transformer?"
        )
    )

    print(
        f"Query embedding dimensions: "
        f"{len(query_vector)}"
    )

    # ---------------------------------------------------------
    # Test LLM
    # ---------------------------------------------------------

    print("\n========================================")
    print("Testing LLM")
    print("========================================")

    llm = loader.load_llm()

    result = llm.invoke(
        "Explain Transformer in one sentence."
    )

    print(
        f"LLM Result: {result.content}"
    )