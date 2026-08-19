"""
Conversational RAG implementation.

IMPORTANT ARCHITECTURE:

                         INGESTION
                            │
                            ▼
                  RETRIEVAL_DOCUMENT
                            │
                            ▼
                          FAISS
                            │
                            │
                    ────────┼────────
                            │
                            ▼
                         RETRIEVAL
                            │
                         Query
                            │
                            ▼
                  RETRIEVAL_QUERY
                            │
                            ▼
                     query vector
                            │
                            ▼
              FAISS.search_by_vector()
                            │
                            ▼
                      Documents
                            │
                            ▼
                         LLM
                            │
                            ▼
                         Answer

The critical difference from the old implementation is that we DO NOT
let FAISS call embedding_function.embed_query() implicitly.

Instead:

    query_vector = query_embeddings.embed_query(query)

followed by:

    vectorstore.similarity_search_by_vector(query_vector)

This makes query embedding completely explicit.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_LLM_MODEL = "gemini-3.6-flash"
DEFAULT_INDEX_NAME = "index"

DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_google_api_key() -> str:

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    return api_key


def _message_to_text(message: Any) -> str:

    if isinstance(message, str):
        return message

    if isinstance(message, BaseMessage):
        content = message.content

        if isinstance(content, str):
            return content

        return str(content)

    return str(message)


def _history_to_text(
    chat_history: Optional[
        Sequence[Union[BaseMessage, Dict[str, Any], str]]
    ],
) -> str:

    if not chat_history:
        return ""

    lines: List[str] = []

    for message in chat_history:

        if isinstance(message, HumanMessage):
            role = "User"

        elif isinstance(message, AIMessage):
            role = "Assistant"

        elif isinstance(message, dict):

            role = str(
                message.get(
                    "role",
                    "User",
                )
            )

        else:
            role = "Message"

        lines.append(
            f"{role}: {_message_to_text(message)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explicit FAISS Retriever
# ---------------------------------------------------------------------------

class ExplicitFAISSRetriever(BaseRetriever):
    """
    FAISS retriever that explicitly performs query embedding.

    We deliberately do NOT use:

        vectorstore.similarity_search(query)

    because that causes FAISS to call its attached embedding object
    internally.

    Instead we do:

        query_vector = embeddings.embed_query(query)

        vectorstore.similarity_search_by_vector(
            query_vector,
            k=k
        )
    """

    vectorstore: Any
    query_embeddings: Any

    k: int = DEFAULT_K

    search_type: str = "similarity"

    fetch_k: int = 20

    lambda_mult: float = 0.5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        if self.vectorstore is None:
            raise RuntimeError(
                "FAISS vectorstore is not initialized."
            )

        if self.query_embeddings is None:
            raise RuntimeError(
                "Query embedding model is not initialized."
            )

        # ==============================================================
        # CRITICAL FIX
        #
        # Explicitly call the QUERY embedding model.
        # ==============================================================
        logger.info(
            "Creating query embedding | query=%s",
            query,
        )

        query_vector = (
            self.query_embeddings.embed_query(
                query
            )
        )

        if not query_vector:
            raise RuntimeError(
                "Query embedding returned an empty vector."
            )

        logger.info(
            "Query embedding successful | dimension=%d",
            len(query_vector),
        )

        # ==============================================================
        # CRITICAL FIX
        #
        # Search FAISS BY VECTOR.
        #
        # FAISS does not need to call embed_query() anymore.
        # ==============================================================

        search_type = (
            self.search_type or "similarity"
        ).lower()

        if search_type == "mmr":

            logger.info(
                "Running FAISS MMR vector retrieval | k=%d | fetch_k=%d | lambda=%s",
                self.k,
                self.fetch_k,
                self.lambda_mult,
            )

            documents = (
                self.vectorstore
                .max_marginal_relevance_search_by_vector(
                    embedding=query_vector,
                    k=self.k,
                    fetch_k=max(
                        self.fetch_k,
                        self.k,
                    ),
                    lambda_mult=self.lambda_mult,
                )
            )

        else:

            logger.info(
                "Running FAISS similarity vector retrieval | k=%d",
                self.k,
            )

            documents = (
                self.vectorstore
                .similarity_search_by_vector(
                    embedding=query_vector,
                    k=self.k,
                )
            )

        logger.info(
            "FAISS retrieval successful | documents=%d",
            len(documents),
        )

        return documents


# ---------------------------------------------------------------------------
# ConversationalRAG
# ---------------------------------------------------------------------------

class ConversationalRAG:
    """
    Conversational RAG using:

    - Gemini LLM
    - Gemini query embeddings
    - FAISS
    - Explicit vector retrieval

    Public API compatible with:

        rag = ConversationalRAG(session_id=session_id)

        rag.load_retriever_from_faiss(...)

        answer = rag.invoke(
            user_input,
            chat_history=chat_history,
        )
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        k: int = DEFAULT_K,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):

        self.session_id = session_id

        self.embedding_model_name = (
            embedding_model
            or os.getenv(
                "GOOGLE_EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            )
        )

        self.llm_model_name = (
            llm_model
            or os.getenv(
                "GOOGLE_LLM_MODEL",
                DEFAULT_LLM_MODEL,
            )
        )

        self.k = k

        self.search_type = search_type

        self.fetch_k = fetch_k

        self.lambda_mult = lambda_mult

        self.query_embeddings: Optional[
            GoogleGenerativeAIEmbeddings
        ] = None

        self.vectorstore: Optional[FAISS] = None

        self.retriever: Optional[
            ExplicitFAISSRetriever
        ] = None

        self.llm: Optional[
            ChatGoogleGenerativeAI
        ] = None

        self.chain = None

        logger.info(
            "ConversationalRAG initialized | session=%s",
            self.session_id,
        )

        self._initialize_query_embeddings()

        self._initialize_llm()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _initialize_query_embeddings(self):

        api_key = _get_google_api_key()

        logger.info(
            "Loading QUERY embedding model | model=%s | task_type=RETRIEVAL_QUERY",
            self.embedding_model_name,
        )

        self.query_embeddings = (
            GoogleGenerativeAIEmbeddings(
                model=self.embedding_model_name,
                task_type="RETRIEVAL_QUERY",
                google_api_key=api_key,
            )
        )

        logger.info(
            "Query embedding model loaded successfully."
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _initialize_llm(self):

        api_key = _get_google_api_key()

        logger.info(
            "Loading Gemini LLM | model=%s",
            self.llm_model_name,
        )

        # DO NOT pass temperature.
        #
        # Gemini versions such as the one in the user's environment
        # report:
        #
        # Model uses fixed sampling defaults;
        # temperature will be ignored.
        #
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            google_api_key=api_key,
        )

        logger.info(
            "Gemini LLM loaded successfully."
        )

    # ------------------------------------------------------------------
    # FAISS loading
    # ------------------------------------------------------------------

    def load_retriever_from_faiss(
        self,
        index_path: Union[str, Path],
        k: Optional[int] = None,
        index_name: Optional[str] = None,
        search_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ):
        """
        Load FAISS and create an ExplicitFAISSRetriever.

        IMPORTANT:

        We pass the query embedding object to FAISS.load_local only
        because the LangChain API requires an embeddings object.

        Actual retrieval DOES NOT use FAISS.similarity_search(query).

        It uses:

            embed_query()
            +
            similarity_search_by_vector()
        """

        if self.query_embeddings is None:
            raise RuntimeError(
                "Query embedding model is not initialized."
            )

        index_path = Path(index_path)

        index_name = (
            index_name
            or os.getenv(
                "FAISS_INDEX_NAME",
                DEFAULT_INDEX_NAME,
            )
        )

        actual_k = (
            k
            if k is not None
            else self.k
        )

        actual_search_type = (
            search_type
            if search_type is not None
            else self.search_type
        )

        actual_fetch_k = (
            fetch_k
            if fetch_k is not None
            else self.fetch_k
        )

        actual_lambda = (
            lambda_mult
            if lambda_mult is not None
            else self.lambda_mult
        )

        faiss_file = (
            index_path
            / f"{index_name}.faiss"
        )

        pickle_file = (
            index_path
            / f"{index_name}.pkl"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS directory does not exist: "
                f"{index_path}"
            )

        if not faiss_file.exists():
            raise FileNotFoundError(
                f"FAISS index file does not exist: "
                f"{faiss_file}"
            )

        if not pickle_file.exists():
            raise FileNotFoundError(
                f"FAISS metadata file does not exist: "
                f"{pickle_file}"
            )

        logger.info(
            "Loading FAISS | path=%s | index=%s",
            index_path,
            index_name,
        )

        # --------------------------------------------------------------
        # Load FAISS.
        #
        # The embedding object is supplied because FAISS.load_local()
        # requires it.
        #
        # BUT:
        # retrieval.py never calls vectorstore.similarity_search().
        # --------------------------------------------------------------

        self.vectorstore = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=self.query_embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )

        logger.info(
            "FAISS loaded successfully."
        )

        # --------------------------------------------------------------
        # Create our explicit retriever.
        # --------------------------------------------------------------

        self.retriever = ExplicitFAISSRetriever(
            vectorstore=self.vectorstore,
            query_embeddings=self.query_embeddings,
            k=actual_k,
            search_type=actual_search_type,
            fetch_k=max(
                actual_fetch_k,
                actual_k,
            ),
            lambda_mult=actual_lambda,
        )

        logger.info(
            "Explicit FAISS retriever initialized | "
            "search_type=%s | k=%d",
            actual_search_type,
            actual_k,
        )

        return self.retriever

    # ------------------------------------------------------------------
    # Query embedding test
    # ------------------------------------------------------------------

    def test_query_embedding(
        self,
        query: str,
    ) -> List[float]:
        """
        Directly test the query embedding.

        This is useful for diagnosing Google embedding failures.

        It does NOT touch FAISS.
        """

        if self.query_embeddings is None:
            raise RuntimeError(
                "Query embedding model is not initialized."
            )

        logger.info(
            "Testing query embedding directly | query=%s",
            query,
        )

        vector = (
            self.query_embeddings.embed_query(
                query
            )
        )

        logger.info(
            "Query embedding test successful | dimension=%d",
            len(vector),
        )

        return vector

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        if self.retriever is None:
            raise RuntimeError(
                "Retriever is not initialized. "
                "Call load_retriever_from_faiss() first."
            )

        if not query or not query.strip():
            return []

        logger.info(
            "Running vector retrieval | query=%s",
            query,
        )

        # IMPORTANT:
        #
        # ExplicitFAISSRetriever.invoke()
        #
        # performs:
        #
        #     embed_query()
        #
        # followed by:
        #
        #     similarity_search_by_vector()
        #
        documents = self.retriever.invoke(
            query
        )

        return documents

    # ------------------------------------------------------------------
    # Query transformation
    # ------------------------------------------------------------------

    def _rewrite_query(
        self,
        user_input: str,
        chat_history: Optional[
            Sequence[
                Union[
                    BaseMessage,
                    Dict[str, Any],
                    str,
                ]
            ]
        ] = None,
    ) -> str:
        """
        Convert a conversational question into a standalone retrieval query.

        For a first question this normally returns the same question.

        For example:

            User:
                What is Agentic AI?

        becomes:

            What is Agentic AI?

        Follow-up:

            User:
                What are its benefits?

        with history can become:

            What are the benefits of Agentic AI?
        """

        if not chat_history:
            return user_input.strip()

        if self.llm is None:
            return user_input.strip()

        history_text = _history_to_text(
            chat_history
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a query rewriting component for a RAG system.

Rewrite the user's latest question into one standalone
search query suitable for semantic document retrieval.

Rules:
- Resolve references using the conversation history.
- Preserve important technical terms.
- Do not answer the question.
- Do not explain the rewrite.
- Return ONLY the rewritten search query.
""",
                ),
                (
                    "human",
                    """
Conversation history:

{history}

Latest user question:

{question}

Standalone retrieval query:
""",
                ),
            ]
        )

        try:

            response = (
                prompt
                | self.llm
            ).invoke(
                {
                    "history": history_text,
                    "question": user_input,
                }
            )

            rewritten = _message_to_text(
                response
            ).strip()

            if rewritten:
                logger.info(
                    "Query rewritten | original=%s | rewritten=%s",
                    user_input,
                    rewritten,
                )

                return rewritten

        except Exception as exc:

            logger.warning(
                "Query rewriting failed. "
                "Using original query. Error=%s",
                exc,
            )

        return user_input.strip()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @staticmethod
    def _format_documents(
        documents: Sequence[Document],
    ) -> str:

        if not documents:
            return "No relevant documents were retrieved."

        context_parts: List[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata or {}

            source = metadata.get(
                "source_file",
                metadata.get(
                    "source",
                    "unknown",
                ),
            )

            page = metadata.get(
                "page",
                "",
            )

            source_label = (
                f"{source}"
                if not page
                else f"{source}, page {page}"
            )

            context_parts.append(
                f"[Document {index} | {source_label}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(
            context_parts
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        user_question: str,
        retrieval_query: str,
        documents: Sequence[Document],
        chat_history: Optional[
            Sequence[
                Union[
                    BaseMessage,
                    Dict[str, Any],
                    str,
                ]
            ]
        ] = None,
    ) -> str:

        if self.llm is None:
            raise RuntimeError(
                "Gemini LLM is not initialized."
            )

        context = self._format_documents(
            documents
        )

        history_text = _history_to_text(
            chat_history
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful RAG assistant.

Answer the user's question using the supplied context.

Rules:

1. Prefer the supplied context over general knowledge.
2. If the answer is present in the context, explain it clearly.
3. If the context does not contain enough information, say:
   "I don't have enough information in the provided documents."
4. Do not invent facts.
5. Do not mention internal retrieval implementation.
6. Keep the answer concise but useful.
7. Use simple examples when they help explain a technical concept.

Retrieved context:

{context}
""",
                ),
                (
                    "human",
                    """
Conversation history:

{history}

User question:

{question}

Retrieval query:

{retrieval_query}

Answer:
""",
                ),
            ]
        )

        response = (
            prompt
            | self.llm
        ).invoke(
            {
                "context": context,
                "history": history_text,
                "question": user_question,
                "retrieval_query": retrieval_query,
            }
        )

        answer = _message_to_text(
            response
        ).strip()

        return answer

    # ------------------------------------------------------------------
    # Public invoke
    # ------------------------------------------------------------------

    def invoke(
        self,
        user_input: str,
        chat_history: Optional[
            Sequence[
                Union[
                    BaseMessage,
                    Dict[str, Any],
                    str,
                ]
            ]
        ] = None,
    ) -> str:

        if not user_input or not user_input.strip():
            return ""

        logger.info(
            "Invoking RAG | session=%s | user_input=%s",
            self.session_id,
            user_input,
        )

        try:

            # ----------------------------------------------------------
            # 1. Rewrite query
            # ----------------------------------------------------------

            retrieval_query = self._rewrite_query(
                user_input=user_input,
                chat_history=chat_history,
            )

            # ----------------------------------------------------------
            # 2. Retrieve documents
            #
            # ExplicitFAISSRetriever:
            #
            #     Gemini RETRIEVAL_QUERY
            #             ↓
            #       query vector
            #             ↓
            #     FAISS by vector
            # ----------------------------------------------------------

            documents = self.retrieve(
                retrieval_query
            )

            logger.info(
                "Retrieved documents | count=%d",
                len(documents),
            )

            # ----------------------------------------------------------
            # 3. Generate answer
            # ----------------------------------------------------------

            answer = self._generate_answer(
                user_question=user_input,
                retrieval_query=retrieval_query,
                documents=documents,
                chat_history=chat_history,
            )

            logger.info(
                "RAG answer generated successfully."
            )

            return answer

        except Exception as exc:

            logger.exception(
                "RAG invocation failed."
            )

            # Do not hide the actual exception.
            raise RuntimeError(
                f"RAG invocation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ):
        """
        Diagnostic method.

        Explicitly embeds the query and performs FAISS
        similarity search with scores.
        """

        if self.vectorstore is None:
            raise RuntimeError(
                "FAISS vectorstore is not initialized."
            )

        if self.query_embeddings is None:
            raise RuntimeError(
                "Query embeddings are not initialized."
            )

        actual_k = (
            k
            if k is not None
            else self.k
        )

        logger.info(
            "Embedding query for score retrieval."
        )

        query_vector = (
            self.query_embeddings.embed_query(
                query
            )
        )

        return (
            self.vectorstore
            .similarity_search_with_score_by_vector(
                embedding=query_vector,
                k=actual_k,
            )
        )