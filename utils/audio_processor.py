import os
import uuid
import shutil
import yt_dlp
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi

BASE_DOWNLOAD_DIR = 'downloads'
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)


def create_job_dir() -> str:
    """Creates a unique folder for this processing job to avoid filename collisions."""
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(BASE_DOWNLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from common URL formats."""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url


# ---------------------------------------------------------------------------
# PRIMARY: yt-dlp with cookies (real audio download -> Groq/Sarvam transcription)
# ---------------------------------------------------------------------------

def _download_with_ytdlp(url: str, job_dir: str) -> str:
    output_path = os.path.join(job_dir, "%(id)s.%(ext)s")

    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils folder
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    cookies_path = os.path.join(project_root, "youtube_cookies.txt")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        "quiet": True,
        "cookiefile": cookies_path if os.path.exists(cookies_path) else None,
        "extractor_args": {
            "youtube": {
                # 'web' actually uses the cookie session; ios/android/mweb largely ignore it
                "player_client": ["web"]
            }
        },
    }

    if os.path.exists(cookies_path):
        print(f"🍪 Found cookie authentication file at {cookies_path}")
    else:
        print("⚠️ No youtube_cookies.txt found — proceeding without authentication.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".mp3"
    return filename


# ---------------------------------------------------------------------------
# FALLBACK: YouTube's own captions via youtube-transcript-api (text only)
# ---------------------------------------------------------------------------

def _download_transcript_fallback(url: str, job_dir: str) -> str:
    video_id = extract_video_id(url)
    print(f"📄 Falling back to YouTube captions for video: {video_id}")

    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "hi", "bn"])

    full_text = " ".join(snippet.text for snippet in fetched)

    transcript_path = os.path.join(job_dir, f"{video_id}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print("✅ Caption-based transcript fetched successfully.")
    return transcript_path


def download_youtube_audio(url: str, job_dir: str) -> str:
    """
    Tries yt-dlp (real audio -> full Whisper/Sarvam transcription) first.
    If that fails (cloud IP blocks, format errors, etc.), falls back to
    YouTube's existing captions via youtube-transcript-api.
    If both fail, raises an error telling the user to upload the file directly.
    """
    try:
        return _download_with_ytdlp(url, job_dir)
    except Exception as e:
        print(f"⚠️ yt-dlp download failed: {e}")
        try:
            return _download_transcript_fallback(url, job_dir)
        except Exception as e2:
            print(f"❌ Caption fallback also failed: {e2}")
            raise RuntimeError(
                "Could not download audio or captions for this YouTube video "
                "(YouTube may be blocking this server). "
                "Please use the 'Upload file' option instead."
            )


# ---------------------------------------------------------------------------
# LOCAL FILE PROCESSING (unchanged)
# ---------------------------------------------------------------------------

def convert_to_mp3(input_path: str, job_dir: str) -> str:
    """Convert any audio/video file to a mono, 16kHz, 64kbps MP3 inside the job folder."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(job_dir, f"{base_name}_converted.mp3")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="64k")
    return output_path


def chunk_audio(audio_path: str, job_dir: str, chunk_minutes: int = 10) -> list:
    """
    Slices audio into chunk_minutes-long MP3 chunks.
    If the input is a .txt (caption fallback), pass it through unchanged.
    """
    if audio_path.endswith(".txt"):
        return [audio_path]

    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = os.path.join(job_dir, f"{base_name}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str):
    """
    Returns (chunks, job_dir).
    job_dir should be passed to cleanup_job() once transcription is done.
    """
    job_dir = create_job_dir()

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        audio_path = download_youtube_audio(source, job_dir)
    else:
        print("Detected local file. Converting to optimized MP3...")
        audio_path = convert_to_mp3(source, job_dir)

    print("Chunking audio...")
    chunks = chunk_audio(audio_path, job_dir)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks, job_dir


def cleanup_job(job_dir: str):
    """Deletes all downloaded/converted/chunked files for this job."""
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        print(f"🧹 Cleaned up temp files in {job_dir}")