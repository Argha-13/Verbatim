import os
import uuid
import shutil
import yt_dlp
from pydub import AudioSegment

BASE_DOWNLOAD_DIR = 'downloads'
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)


def create_job_dir() -> str:
    """Creates a unique folder for this processing job to avoid filename collisions."""
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(BASE_DOWNLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def download_youtube_audio(url: str, job_dir: str) -> str:
    output_path = os.path.join(job_dir, "%(id)s.%(ext)s")
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
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android", "mweb"]
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".mp3"
    return filename


def convert_to_mp3(input_path: str, job_dir: str) -> str:
    """Convert any audio/video file to a mono, 16kHz, 64kbps MP3 inside the job folder."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(job_dir, f"{base_name}_converted.mp3")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="64k")
    return output_path


def chunk_audio(audio_path: str, job_dir: str, chunk_minutes: int = 10) -> list:
    """Slices audio into chunk_minutes-long MP3 chunks inside the job folder."""
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