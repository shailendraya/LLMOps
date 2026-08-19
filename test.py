import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
)

from multi_doc_chat.src.document_chat.retrieval import (
    ConversationalRAG,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# TEST
# =========================================================

def test_document_ingestion_and_rag():

    uploaded_files = []

    try:

        # -------------------------------------------------
        # Test documents
        # -------------------------------------------------

        test_files = [
            "./data/NIPS-2017-attention-is-all-you-need-Paper.pdf",
            "./data/AgenticAI.txt",
        ]

        # -------------------------------------------------
        # Validate files
        # -------------------------------------------------

        for file_path in test_files:

            path = Path(file_path)

            if not path.exists():

                print(
                    f"File does not exist: {file_path}"
                )

                continue

            uploaded_files.append(
                open(
                    path,
                    "rb",
                )
            )

        if not uploaded_files:

            print(
                "No valid files to upload."
            )

            sys.exit(1)

        # =================================================
        # STEP 1
        # INGEST DOCUMENTS
        # =================================================

        print(
            "\n========================================"
        )
        print(
            "STEP 1: DOCUMENT INGESTION"
        )
        print(
            "========================================"
        )

        ci = ChatIngestor(
            temp_base="data",
            faiss_base="faiss_index",
            use_session_dirs=True,
        )

        print(
            f"Session ID: {ci.session_id}"
        )

        # -------------------------------------------------
        # Build FAISS retriever
        # -------------------------------------------------

        ci.built_retriver(
            uploaded_files,
            chunk_size=1000,
            chunk_overlap=150,
            k=5,
            search_type="similarity",
            fetch_k=20,
            lambda_mult=0.5,
        )

        print(
            "FAISS index created successfully."
        )

        # -------------------------------------------------
        # Close uploaded files
        # -------------------------------------------------

        for file in uploaded_files:

            try:
                file.close()
            except Exception:
                pass

        uploaded_files.clear()

        # =================================================
        # STEP 2
        # LOAD RAG
        # =================================================

        print(
            "\n========================================"
        )
        print(
            "STEP 2: LOAD CONVERSATIONAL RAG"
        )
        print(
            "========================================"
        )

        session_id = ci.session_id

        index_dir = os.path.join(
            "faiss_index",
            session_id,
        )

        rag = ConversationalRAG(
            session_id=session_id,
        )

        rag.load_retriever_from_faiss(
            index_path=index_dir,
            k=5,
            index_name=os.getenv(
                "FAISS_INDEX_NAME",
                "index",
            ),
            search_type="similarity",
        )

        print(
            "Conversational RAG loaded successfully."
        )

        # =================================================
        # STEP 3
        # INTERACTIVE CHAT
        # =================================================

        print(
            "\n========================================"
        )
        print(
            "STEP 3: INTERACTIVE CHAT"
        )
        print(
            "========================================"
        )

        print(
            "\nType 'exit' to quit.\n"
        )

        chat_history = []

        while True:

            try:

                user_input = input(
                    "You: "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                print(
                    "\nExiting chat."
                )

                break

            # -------------------------------------------------
            # Empty input
            # -------------------------------------------------

            if not user_input:
                continue

            # -------------------------------------------------
            # Exit
            # -------------------------------------------------

            if user_input.lower() in {
                "exit",
                "quit",
                "q",
                ":q",
            }:

                print(
                    "Goodbye!"
                )

                break

            # =================================================
            # INVOKE RAG
            # =================================================

            try:

                answer = rag.invoke(
                    user_input,
                    chat_history=chat_history,
                )

                print(
                    f"\nAssistant: {answer}\n"
                )

                # -------------------------------------------------
                # Maintain conversation history
                # -------------------------------------------------

                chat_history.append(
                    HumanMessage(
                        content=user_input
                    )
                )

                chat_history.append(
                    AIMessage(
                        content=answer
                    )
                )

            except Exception as e:

                print(
                    "\nRAG invocation failed:"
                )

                print(
                    str(e)
                )

                print()

        print(
            "\nTest completed successfully."
        )

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "TEST FAILED"
        )

        print(
            "========================================"
        )

        print(
            str(e)
        )

        sys.exit(1)

    finally:

        # -------------------------------------------------
        # Always close file handles
        # -------------------------------------------------

        for file in uploaded_files:

            try:
                file.close()
            except Exception:
                pass


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    test_document_ingestion_and_rag()