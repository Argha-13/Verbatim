import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _source_label(content_type: str) -> str:
    return "meeting transcript" if content_type.lower() == "meeting" else "video transcript"


def _build_prompt(content_type: str) -> ChatPromptTemplate:
    label = _source_label(content_type)
    return ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are an expert assistant for analyzing a {label}.
            Answer the user's question based ONLY on the {label} context provided below.

            If the answer is not found in the context, say:
            "I could not find this information in the transcript."

            Always be concise and precise. If quoting someone, mention it clearly.

            Context from {label}:
            {{context}}""",
        ),
        ("human", "{question}"),
    ])


def build_rag_chain(transcript: str, collection_name: str, content_type: str = "meeting"):
    vector_store = build_vector_store(transcript, collection_name)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = _build_prompt(content_type)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain(collection_name: str, content_type: str = "meeting"):
    vector_store = load_vector_store(collection_name)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    prompt = _build_prompt(content_type)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)