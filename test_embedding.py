from multi_doc_chat.utils.model_loader import ModelLoader


def main():

    loader = ModelLoader()

    query_embeddings = loader.load_query_embeddings()

    print("Embedding model:", query_embeddings)

    text = "What is Agentic AI?"

    print("\nTesting query embedding...")

    vector = query_embeddings.embed_query(text)

    print("SUCCESS")
    print("Vector dimension:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    main()