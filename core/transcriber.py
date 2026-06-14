import os
import time
from pydub import AudioSegment

# Sarvam sync REST API only accepts files <= 30 seconds.
SARVAM_PIECE_SECONDS = 25
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry


# ---------------- LAZY CLIENT INIT ----------------

_groq_client = None
_sarvam_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq()  # picks up GROQ_API_KEY from environment
    return _groq_client


def _get_sarvam_client():
    global _sarvam_client
    if _sarvam_client is None:
        from sarvamai import SarvamAI
        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
        _sarvam_client = SarvamAI(api_subscription_key=api_key)
    return _sarvam_client


# ---------------- RETRY HELPER ----------------

def _with_retries(func, *args, label: str = "API call", **kwargs):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"⚠️ {label} failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ {label} failed after {MAX_RETRIES} attempts: {e}")
    raise last_err


# ---------------- GROQ ----------------

def transcribe_chunk_groq(chunk_path: str) -> str:
    """Sends the optimized MP3 chunk directly to Groq's Hosted Whisper API."""
    print(" Sending chunk to Groq Whisper Cloud...")

    def _call():
        client = _get_groq_client()
        with open(chunk_path, "rb") as file:
            translation = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        return str(translation)

    return _with_retries(_call, label=f"Groq transcription ({os.path.basename(chunk_path)})")


# ---------------- SARVAM ----------------

def _send_to_sarvam_sdk(piece_path: str) -> str:
    """Sends one <=30s file to Sarvam via official SDK and gets the transcript."""

    def _call():
        client = _get_sarvam_client()
        with open(piece_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model=SARVAM_MODEL,
                mode="translate"  # Translates Hinglish/Indic audio into English text natively
            )
        return response.transcript if hasattr(response, 'transcript') else ""

    return _with_retries(_call, label=f"Sarvam transcription ({os.path.basename(piece_path)})")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam REST API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each via the SDK, and join the transcripts.
    """
    audio = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    pieces_text = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.mp3"
        piece.export(piece_path, format="mp3", bitrate="64k")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            text = _send_to_sarvam_sdk(piece_path)
            pieces_text.append(text)
        except Exception:
            # Mark the failure explicitly instead of silently dropping audio
            pieces_text.append("[transcription failed for this segment]")
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(pieces_text).strip()


# ---------------- ROUTER ----------------

SARVAM_LANGUAGES = {"hinglish", "benglish", "bengali", "hindi"}


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Groq Cloud or Sarvam depending on language choice.
    - english  → Groq Cloud (Whisper API)
    - hinglish / benglish / bengali / hindi → Sarvam (translates Indic audio to English)
    """
    lang = language.lower()

    if lang in SARVAM_LANGUAGES:
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_groq(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "Sarvam AI SDK" if language.lower() in SARVAM_LANGUAGES else "Groq Hosted Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()