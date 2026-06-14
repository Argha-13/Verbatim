import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


# Prompts differ slightly depending on whether the source is a meeting
# (decision/action oriented) or a YouTube/general video (topic/insight oriented).
MAP_PROMPTS = {
    "meeting": "Summarize this portion of a meeting transcript concisely, "
               "focusing on what was discussed, decided, and any action items.",
    "youtube": "Summarize this portion of a video transcript concisely, "
               "focusing on the key points, topics, and insights covered.",
}

COMBINE_PROMPTS = {
    "meeting": "You are an expert meeting summarizer. Combine these partial summaries "
               "into one final professional meeting summary in bullet points.",
    "youtube": "You are an expert content summarizer. Combine these partial summaries "
               "into one final, well-organized summary of the video in bullet points, "
               "covering the main topics and takeaways.",
}


def summarize(transcript: str, content_type: str = "meeting") -> str:
    llm = get_llm()
    content_type = content_type.lower()

    map_prompt_text = MAP_PROMPTS.get(content_type, MAP_PROMPTS["meeting"])
    combine_prompt_text = COMBINE_PROMPTS.get(content_type, COMBINE_PROMPTS["meeting"])

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", map_prompt_text),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Run all chunk summaries in parallel instead of sequentially
    chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", combine_prompt_text),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the transcript, generate a short professional title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])