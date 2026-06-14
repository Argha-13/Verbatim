import os
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

CHROMA_DIR = "vector_db"


def get_embeddings():
    """
    Returns Mistral's hosted embedding model (mistral-embed).
    Uses the same MISTRAL_API_KEY already set for the LLM calls.
    Free tier covers this easily for typical transcript sizes.
    """
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def build_vector_store(transcript: str, collection_name: str) -> Chroma:
    print(f"Building Vector Store (collection: {collection_name})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store(collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )


def delete_collection(collection_name: str):
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(name=collection_name)
        print(f"🧹 Deleted vector store collection: {collection_name}")
    except Exception as e:
        print(f"⚠️ Could not delete collection {collection_name}: {e}")


def list_collections() -> list:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return [c.name for c in client.list_collections()]