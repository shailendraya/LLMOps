"""
Document ingestion and FAISS index management.

Architecture:

Documents
    ↓
Load
    ↓
Split
    ↓
Gemini Embeddings - RETRIEVAL_DOCUMENT
    ↓
FAISS
    ↓
Save index

IMPORTANT:
This module is responsible ONLY for document embeddings.

Query embeddings are created separately in retrieval.py using:
    task_type="RETRIEVAL_QUERY"
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable, List, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
DEFAULT_INDEX_NAME = "index"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_google_api_key() -> str:
    """
    Get Google API key.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not configured.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. "
            "Please add GOOGLE_API_KEY to your .env file."
        )

    return api_key


def _safe_filename(filename: str) -> str:
    """
    Create a safe filename.
    """

    filename = Path(filename).name

    if not filename:
        filename = "uploaded_file"

    return filename


def _file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _unique_filename(filename: str) -> str:
    """
    Generate a unique filename while retaining the extension.
    """

    extension = Path(filename).suffix.lower()

    return f"{uuid.uuid4().hex}{extension}"


# ---------------------------------------------------------------------------
# ChatIngestor
# ---------------------------------------------------------------------------

class ChatIngestor:
    """
    Handles document ingestion and FAISS creation.

    Example:

        ci = ChatIngestor(
            temp_base="data",
            faiss_base="faiss_index",
            use_session_dirs=True,
        )

        retriever = ci.built_retriver(
            uploaded_files,
            chunk_size=1000,
            chunk_overlap=150,
            k=5,
            search_type="similarity",
        )
    """

    def __init__(
        self,
        temp_base: Union[str, Path] = "data",
        faiss_base: Union[str, Path] = "faiss_index",
        use_session_dirs: bool = True,
        embedding_model: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        self.temp_base = Path(temp_base)
        self.faiss_base = Path(faiss_base)

        self.use_session_dirs = use_session_dirs

        self.embedding_model_name = (
            embedding_model
            or os.getenv(
                "GOOGLE_EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            )
        )

        self.index_name = (
            index_name
            or os.getenv(
                "FAISS_INDEX_NAME",
                DEFAULT_INDEX_NAME,
            )
        )

        self.session_id = self._create_session_id()

        if self.use_session_dirs:
            self.temp_dir = self.temp_base / self.session_id
            self.faiss_dir = self.faiss_base / self.session_id
        else:
            self.temp_dir = self.temp_base
            self.faiss_dir = self.faiss_base

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        self.document_embeddings: Optional[
            GoogleGenerativeAIEmbeddings
        ] = None

        self.vectorstore: Optional[FAISS] = None

        logger.info(
            "ChatIngestor initialized | session=%s | temp=%s | faiss=%s",
            self.session_id,
            self.temp_dir,
            self.faiss_dir,
        )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _create_session_id(self) -> str:
        """
        Create a session ID.
        """

        import datetime

        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        random_part = uuid.uuid4().hex[:8]

        return f"session_{timestamp}_{random_part}"

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _save_uploaded_file(
        self,
        uploaded_file: Union[BinaryIO, str, Path],
    ) -> Path:
        """
        Save an uploaded file into the session directory.

        Supports:

        - file-like object
        - string path
        - pathlib.Path
        """

        # --------------------------------------------------------------
        # File path
        # --------------------------------------------------------------

        if isinstance(uploaded_file, (str, Path)):
            source_path = Path(uploaded_file)

            if not source_path.exists():
                raise FileNotFoundError(
                    f"File does not exist: {source_path}"
                )

            filename = _safe_filename(source_path.name)
            destination = self.temp_dir / _unique_filename(filename)

            shutil.copy2(source_path, destination)

            logger.info(
                "File saved | source=%s | destination=%s",
                source_path,
                destination,
            )

            return destination

        # --------------------------------------------------------------
        # File-like object
        # --------------------------------------------------------------

        filename = getattr(
            uploaded_file,
            "name",
            "uploaded_file",
        )

        filename = _safe_filename(filename)

        destination = self.temp_dir / _unique_filename(filename)

        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        with open(destination, "wb") as output_file:

            while True:

                chunk = uploaded_file.read(1024 * 1024)

                if not chunk:
                    break

                output_file.write(chunk)

        logger.info(
            "Uploaded file saved | destination=%s",
            destination,
        )

        return destination

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load one file.
        """

        extension = _file_extension(file_path.name)

        logger.info(
            "Loading file | %s",
            file_path,
        )

        if extension == ".pdf":

            loader = PyPDFLoader(str(file_path))

            documents = loader.load()

        elif extension in {
            ".txt",
            ".md",
            ".markdown",
            ".text",
        }:

            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
                autodetect_encoding=True,
            )

            documents = loader.load()

        else:

            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: PDF, TXT, MD."
            )

        logger.info(
            "Loaded %d documents from %s",
            len(documents),
            file_path.name,
        )

        # Add useful metadata.
        for document in documents:

            document.metadata = {
                **document.metadata,
                "source_file": file_path.name,
                "session_id": self.session_id,
            }

        return documents

    def load_documents(
        self,
        uploaded_files: Sequence[
            Union[BinaryIO, str, Path]
        ],
    ) -> List[Document]:
        """
        Save and load uploaded files.
        """

        if not uploaded_files:
            raise ValueError("No files supplied for ingestion.")

        saved_files: List[Path] = []

        for uploaded_file in uploaded_files:

            saved_path = self._save_uploaded_file(
                uploaded_file
            )

            saved_files.append(saved_path)

        logger.info(
            "Files saved for ingestion | count=%d",
            len(saved_files),
        )

        documents: List[Document] = []

        for file_path in saved_files:

            loaded = self._load_file(file_path)

            documents.extend(loaded)

        if not documents:
            raise RuntimeError(
                "No documents were loaded from the uploaded files."
            )

        self.documents = documents

        logger.info(
            "Documents loaded | count=%d",
            len(documents),
        )

        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def split_documents(
        self,
        documents: Optional[Sequence[Document]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Document]:
        """
        Split documents into chunks.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        source_documents = list(
            documents
            if documents is not None
            else self.documents
        )

        if not source_documents:
            raise RuntimeError(
                "No documents available for splitting."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            source_documents
        )

        if not chunks:
            raise RuntimeError(
                "Document splitting produced zero chunks."
            )

        # Add chunk IDs.
        for index, chunk in enumerate(chunks):

            chunk.metadata = {
                **chunk.metadata,
                "chunk_id": index,
            }

        self.chunks = chunks

        logger.info(
            "Documents split | chunks=%d | chunk_size=%d | overlap=%d",
            len(chunks),
            chunk_size,
            chunk_overlap,
        )

        return chunks

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _create_document_embeddings(
        self,
    ) -> GoogleGenerativeAIEmbeddings:
        """
        Create the DOCUMENT embedding object.

        IMPORTANT:
        This object uses RETRIEVAL_DOCUMENT.

        It must NOT be reused for query retrieval.
        """

        api_key = _get_google_api_key()

        logger.info(
            "Creating document embedding model | model=%s | task_type=RETRIEVAL_DOCUMENT",
            self.embedding_model_name,
        )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_name,
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=api_key,
        )

        self.document_embeddings = embeddings

        return embeddings

    # ------------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------------

    def create_faiss_index(
        self,
        chunks: Optional[Sequence[Document]] = None,
    ) -> FAISS:
        """
        Create a new FAISS index.

        IMPORTANT:
        FAISS is created HERE.

        We do not call methods that expect self.vectorstore
        to exist before this point.
        """

        source_chunks = list(
            chunks
            if chunks is not None
            else self.chunks
        )

        if not source_chunks:
            raise RuntimeError(
                "No chunks available to create FAISS index."
            )

        if self.document_embeddings is None:

            self._create_document_embeddings()

        assert self.document_embeddings is not None

        logger.info(
            "Creating FAISS index | chunks=%d",
            len(source_chunks),
        )

        # --------------------------------------------------------------
        # THIS IS THE IMPORTANT PART
        #
        # Document embedding happens here.
        # FAISS is created directly from the documents.
        # --------------------------------------------------------------

        self.vectorstore = FAISS.from_documents(
            documents=source_chunks,
            embedding=self.document_embeddings,
        )

        if self.vectorstore is None:
            raise RuntimeError(
                "FAISS.from_documents() returned None."
            )

        logger.info(
            "FAISS index created successfully."
        )

        return self.vectorstore

    def save_faiss_index(
        self,
        vectorstore: Optional[FAISS] = None,
        index_path: Optional[Union[str, Path]] = None,
        index_name: Optional[str] = None,
    ) -> Path:
        """
        Save FAISS index.
        """

        store = vectorstore or self.vectorstore

        if store is None:
            raise RuntimeError(
                "FAISS vectorstore is not initialized."
            )

        target_dir = Path(
            index_path
            if index_path is not None
            else self.faiss_dir
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_index_name = (
            index_name or self.index_name
        )

        logger.info(
            "Saving FAISS index | path=%s | name=%s",
            target_dir,
            target_index_name,
        )

        store.save_local(
            folder_path=str(target_dir),
            index_name=target_index_name,
        )

        logger.info(
            "FAISS index saved successfully."
        )

        return target_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def built_retriver(
        self,
        uploaded_files: Sequence[
            Union[BinaryIO, str, Path]
        ],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        k: int = 5,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        """
        Build and save the FAISS index.

        NOTE:
        The returned retriever is useful for immediate retrieval,
        but the primary persisted artifact is the FAISS index.

        Query retrieval in ConversationalRAG uses a separate
        RETRIEVAL_QUERY embedding model.
        """

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        search_type = (
            search_type or "similarity"
        ).lower()

        if search_type not in {
            "similarity",
            "mmr",
        }:
            raise ValueError(
                "search_type must be either "
                "'similarity' or 'mmr'."
            )

        # --------------------------------------------------------------
        # 1. Load documents
        # --------------------------------------------------------------

        documents = self.load_documents(
            uploaded_files
        )

        # --------------------------------------------------------------
        # 2. Split documents
        # --------------------------------------------------------------

        chunks = self.split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # --------------------------------------------------------------
        # 3. Create document embeddings + FAISS
        # --------------------------------------------------------------

        vectorstore = self.create_faiss_index(
            chunks
        )

        # --------------------------------------------------------------
        # 4. Save FAISS
        # --------------------------------------------------------------

        self.save_faiss_index(
            vectorstore=vectorstore
        )

        # --------------------------------------------------------------
        # 5. Return retriever for immediate use
        #
        # This is NOT used by ConversationalRAG.
        # ConversationalRAG explicitly creates a QUERY embedding.
        # --------------------------------------------------------------

        if search_type == "mmr":

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": max(fetch_k, k),
                    "lambda_mult": lambda_mult,
                },
            )

        else:

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                },
            )

        logger.info(
            "Retriever built successfully | search_type=%s | k=%d",
            search_type,
            k,
        )

        return retriever

    # Keep compatibility with possible callers using the
    # correctly spelled method.
    def build_retriever(
        self,
        uploaded_files,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        k=5,
        search_type="similarity",
        fetch_k=20,
        lambda_mult=0.5,
    ):
        return self.built_retriver(
            uploaded_files=uploaded_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )


# ---------------------------------------------------------------------------
# Optional FAISS manager
# ---------------------------------------------------------------------------

class FAISSManager:
    """
    Small compatibility wrapper around FAISS.

    This class intentionally keeps document and query embeddings separate.
    """

    def __init__(
        self,
        index_dir: Union[str, Path],
        embedding_model: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)

        self.embedding_model_name = (
            embedding_model
            or os.getenv(
                "GOOGLE_EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            )
        )

        self.index_name = (
            index_name
            or os.getenv(
                "FAISS_INDEX_NAME",
                DEFAULT_INDEX_NAME,
            )
        )

        self.vectorstore: Optional[FAISS] = None

    def load(
        self,
        query_embeddings: GoogleGenerativeAIEmbeddings,
    ) -> FAISS:
        """
        Load FAISS.

        query_embeddings is accepted for compatibility, but actual query
        retrieval is performed explicitly using embed_query() and
        similarity_search_by_vector() in retrieval.py.
        """

        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index directory does not exist: "
                f"{self.index_dir}"
            )

        faiss_file = (
            self.index_dir
            / f"{self.index_name}.faiss"
        )

        pickle_file = (
            self.index_dir
            / f"{self.index_name}.pkl"
        )

        if not faiss_file.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {faiss_file}"
            )

        if not pickle_file.exists():
            raise FileNotFoundError(
                f"FAISS metadata file not found: {pickle_file}"
            )

        logger.info(
            "Loading FAISS index | %s",
            self.index_dir,
        )

        # The embedding object passed here is intentionally NOT relied
        # upon for query retrieval. retrieval.py explicitly embeds the
        # query and calls similarity_search_by_vector().
        self.vectorstore = FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=query_embeddings,
            index_name=self.index_name,
            allow_dangerous_deserialization=True,
        )

        logger.info(
            "FAISS index loaded successfully."
        )

        return self.vectorstore

    def load_or_create(
        self,
        chunks: Optional[Sequence[Document]] = None,
        document_embeddings: Optional[
            GoogleGenerativeAIEmbeddings
        ] = None,
        query_embeddings: Optional[
            GoogleGenerativeAIEmbeddings
        ] = None,
    ) -> FAISS:
        """
        Compatibility method.

        If index exists -> load it.

        Otherwise -> create it from chunks.

        IMPORTANT:
        document_embeddings are used only when creating an index.
        """

        faiss_file = (
            self.index_dir
            / f"{self.index_name}.faiss"
        )

        pickle_file = (
            self.index_dir
            / f"{self.index_name}.pkl"
        )

        if (
            faiss_file.exists()
            and pickle_file.exists()
        ):

            if query_embeddings is None:
                raise ValueError(
                    "query_embeddings is required "
                    "when loading an existing FAISS index."
                )

            return self.load(
                query_embeddings=query_embeddings
            )

        if not chunks:
            raise ValueError(
                "chunks are required to create a new FAISS index."
            )

        if document_embeddings is None:
            document_embeddings = (
                GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model_name,
                    task_type="RETRIEVAL_DOCUMENT",
                    google_api_key=_get_google_api_key(),
                )
            )

        logger.info(
            "Creating new FAISS index."
        )

        self.vectorstore = FAISS.from_documents(
            documents=list(chunks),
            embedding=document_embeddings,
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vectorstore.save_local(
            folder_path=str(self.index_dir),
            index_name=self.index_name,
        )

        logger.info(
            "New FAISS index created and saved."
        )

        return self.vectorstore
        