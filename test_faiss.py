import os

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
)

from multi_doc_chat.utils.model_loader import (
    ModelLoader,
)


def main():

    loader = ModelLoader()

    print("\n1. Testing QUERY embedding")
    
    query_embeddings = (
        loader.load_query_embeddings()
    )

    query_vector = query_embeddings.embed_query(
        "What is Agentic AI?"
    )

    print(
        "Query vector dimension:",
        len(query_vector),
    )

    print("\n2. Testing DOCUMENT embedding")

    document_embeddings = (
        loader.load_document_embeddings()
    )

    document_vectors = (
        document_embeddings.embed_documents(
            [
                "Agentic AI is an AI system that can reason, plan and take actions.",
                "Transformers are neural network architectures used in modern NLP.",
            ]
        )
    )

    print(
        "Document vectors:",
        len(document_vectors),
    )

    print(
        "Document vector dimension:",
        len(document_vectors[0]),
    )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()