from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


ALLOWED_CONTENT_TYPES = {"meeting", "youtube"}
ALLOWED_LANGUAGES = {"english", "hinglish", "benglish", "bengali", "hindi"}


class ProcessYouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    content_type: str = Field(default="meeting")
    language: str = Field(default="english")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError("URL does not look like a YouTube link")
        return v

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ALLOWED_CONTENT_TYPES:
            raise ValueError(f"content_type must be one of {ALLOWED_CONTENT_TYPES}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"language must be one of {ALLOWED_LANGUAGES}")
        return v


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    job_id: str
    title: str
    summary: str
    content_type: str
    insights: dict
    transcript: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class ChatResponse(BaseModel):
    job_id: str
    question: str
    answer: str