from dotenv import load_dotenv
load_dotenv()
import os
import uuid
from utils.audio_processor import process_input, cleanup_job
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_insights
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, content_type: str = "meeting", language: str = "english") -> dict:
    """
    Runs the full pipeline: download/convert -> transcribe -> summarize ->
    extract insights -> build RAG chain.

    Args:
        source: YouTube URL or local file path.
        content_type: "meeting" or "youtube" — controls which insights are
            extracted and how the RAG/summary prompts are framed.
        language: "english", "hinglish", "benglish", "bengali", or "hindi".

    Returns:
        dict with keys: job_id, title, transcript, summary, insights
        (content_type-specific), content_type, rag_chain.

    Raises:
        Exception if any pipeline step fails (download, transcription, etc.)
        — caller (CLI or UI) is responsible for catching and reporting this.
    """
    content_type = content_type.lower().strip()
    language = language.lower().strip()

    job_id = uuid.uuid4().hex
    job_dir = None

    print(f"Starting AI Video Assistant (job_id: {job_id})")

    try:
        chunks, job_dir = process_input(source)

        transcript = transcribe_all(chunks, language)
        print(f"Raw transcription (first 300 characters): {transcript[:300]}")

        title = generate_title(transcript)
        summary = summarize(transcript, content_type)
        insights = extract_insights(transcript, content_type)

        rag_chain = build_rag_chain(transcript, collection_name=job_id, content_type=content_type)

        return {
            "job_id": job_id,
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "insights": insights,        # dict, keys depend on content_type
            "content_type": content_type,
            "rag_chain": rag_chain,
        }

    finally:
        # Clean up downloaded/converted/chunked audio files regardless of success/failure
        if job_dir:
            cleanup_job(job_dir)


def print_results(result: dict):
    """CLI helper to pretty-print pipeline results."""
    content_type = result["content_type"]
    insights = result["insights"]

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")

    if content_type == "youtube":
        print(f"\n🗂️ Key Topics:\n{insights['key_topics']}")
        print(f"\n💡 Takeaways:\n{insights['takeaways']}")
        print(f"\n💬 Notable Quotes:\n{insights['notable_quotes']}")
    else:
        print(f"\n✅ Action Items:\n{insights['action_items']}")
        print(f"\n🔑 Key Decisions:\n{insights['key_decisions']}")
        print(f"\n❓ Open Questions:\n{insights['open_questions']}")

    print("=" * 60)


if __name__ == "__main__":
    # ---- CLI entry point (for local testing) ----
    source = input("Enter YouTube URL or local file path: ").strip()

    content_type = input("Content type (meeting/youtube): ").strip().lower() or "meeting"
    if content_type not in ("meeting", "youtube"):
        print(f"Unrecognized content type '{content_type}', defaulting to 'meeting'.")
        content_type = "meeting"

    language = input("Language (english/hinglish/benglish/bengali/hindi): ").strip().lower() or "english"

    try:
        result = run_pipeline(source, content_type, language)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise SystemExit(1)

    print_results(result)

    # ---- Phase 2 — Chat with your transcript via RAG ----
    print("\n💬 Chat with your transcript (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        try:
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
        except Exception as e:
            print(f"\n❌ Failed to answer: {e}\n")
