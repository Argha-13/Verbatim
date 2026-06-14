from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def _map_reduce_extract(transcript: str, map_prompt: str, combine_prompt: str) -> str:
    """
    Shared map-reduce pattern: split transcript -> run map_prompt on each chunk
    in parallel -> combine all chunk results with combine_prompt.
    """
    llm = get_llm()
    chunks = split_transcript(transcript)

    map_chain = (
        ChatPromptTemplate.from_messages([
            ("system", map_prompt),
            ("human", "{text}")
        ])
        | llm
        | StrOutputParser()
    )

    chunk_results = map_chain.batch([{"text": chunk} for chunk in chunks])

    combine_chain = (
        ChatPromptTemplate.from_messages([
            ("system", combine_prompt),
            ("human", "{text}")
        ])
        | llm
        | StrOutputParser()
    )

    combined_text = "\n\n".join(chunk_results)
    return combine_chain.invoke({"text": combined_text})


# ================= MEETING MODE =================

def extract_action_items(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "You are an expert meeting analyst. "
            "Extract all action items from this transcript chunk.\n\n"
            "For each provide:\n"
            "- Task description\n"
            "- Owner\n"
            "- Deadline (if mentioned else 'Not specified')\n\n"
            "Format as numbered list. "
            "If none found say 'No action items found.'"
        ),
        combine_prompt=(
            "You are an expert meeting analyst. "
            "Combine all extracted action items into a single clean list. "
            "Remove duplicates and merge repeated items. "
            "Format properly as a numbered list. "
            "If no valid items exist say 'No action items found.'"
        ),
    )


def extract_key_decisions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "You are an expert meeting analyst. "
            "Extract all key decisions made in this transcript chunk. "
            "Format as numbered list. "
            "If none found say 'No key decisions found.'"
        ),
        combine_prompt=(
            "Combine all key decisions into one clean numbered list. "
            "Remove duplicates. "
            "If none found say 'No key decisions found.'"
        ),
    )


def extract_questions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "Extract unresolved questions or follow-up topics "
            "from this transcript chunk. "
            "Format as numbered list. "
            "If none found say 'No open questions found.'"
        ),
        combine_prompt=(
            "Combine all unresolved questions into one clean numbered list. "
            "Remove duplicates. "
            "If none found say 'No open questions found.'"
        ),
    )


# ================= YOUTUBE / VIDEO MODE =================

def extract_key_topics(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "You are an expert content analyst. "
            "Extract the main topics or subjects discussed in this transcript chunk. "
            "Format as a numbered list with short labels. "
            "If none found say 'No distinct topics found.'"
        ),
        combine_prompt=(
            "Combine these topic lists into one clean, deduplicated numbered list "
            "covering the main topics of the entire video, in roughly the order "
            "they were discussed. "
            "If none found say 'No distinct topics found.'"
        ),
    )


def extract_takeaways(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "You are an expert content analyst. "
            "Extract the key takeaways, insights, or lessons from this transcript chunk. "
            "Format as a numbered list. "
            "If none found say 'No key takeaways found.'"
        ),
        combine_prompt=(
            "Combine these takeaways into one clean, deduplicated numbered list "
            "of the most important insights from the entire video. "
            "If none found say 'No key takeaways found.'"
        ),
    )


def extract_notable_quotes(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        map_prompt=(
            "Extract any notable, memorable, or quotable statements from this "
            "transcript chunk, word-for-word as spoken. "
            "Format as a numbered list. "
            "If none found say 'No notable quotes found.'"
        ),
        combine_prompt=(
            "Combine these quotes into one clean, deduplicated numbered list "
            "of the most notable quotes from the entire video. "
            "If none found say 'No notable quotes found.'"
        ),
    )


# ================= DISPATCHER =================

def extract_insights(transcript: str, content_type: str = "meeting") -> dict:
    """
    Returns a dict of extracted insights, with fields depending on content_type:
      - "meeting": action_items, key_decisions, open_questions
      - "youtube": key_topics, takeaways, notable_quotes
    """
    content_type = content_type.lower()

    if content_type == "youtube":
        return {
            "key_topics": extract_key_topics(transcript),
            "takeaways": extract_takeaways(transcript),
            "notable_quotes": extract_notable_quotes(transcript),
        }

    # default: meeting
    return {
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "open_questions": extract_questions(transcript),
    }