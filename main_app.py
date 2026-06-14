"""
Streamlit frontend for the AI Video/Meeting Assistant.
Talks to the FastAPI backend (server.py) over HTTP.

Run with: streamlit run app.py
Make sure the FastAPI backend is running first, e.g.:
    uvicorn app.server:app --reload
"""

import time
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BACKEND_URL = "https://verbatim-backend-ez6y.onrender.com"

st.set_page_config(
    page_title="Verbatim",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0B0E14;
        --surface: #151A23;
        --border: #2D333B;
        --text: #E6EDF3;
        --muted: #8B949E;
        --accent: #5EEAD4;
        --accent-soft: rgba(94, 234, 212, 0.12);
        --amber: #F59E0B;
    }

    /* Base */
    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }

    /* Hide default Streamlit chrome (keep header for sidebar toggle) */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    /* Cards */
    .card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent);
        color: #0B0E14;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.85;
        color: #0B0E14;
    }

    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stFileUploader {
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        color: var(--muted);
        padding: 0.6rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* Monospace blocks for transcript / job ids */
    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--muted);
        background-color: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
    }

    /* Eyebrow label */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.25rem;
    }

    /* Waveform header animation */
    .waveform {
        display: flex;
        align-items: center;
        gap: 4px;
        height: 32px;
    }
    .waveform span {
        display: block;
        width: 4px;
        background-color: var(--accent);
        border-radius: 2px;
        animation: wave 1.2s ease-in-out infinite;
    }
    .waveform span:nth-child(1) { height: 10px; animation-delay: 0s; }
    .waveform span:nth-child(2) { height: 22px; animation-delay: 0.1s; }
    .waveform span:nth-child(3) { height: 32px; animation-delay: 0.2s; }
    .waveform span:nth-child(4) { height: 18px; animation-delay: 0.3s; }
    .waveform span:nth-child(5) { height: 26px; animation-delay: 0.4s; }
    .waveform span:nth-child(6) { height: 12px; animation-delay: 0.5s; }

    @keyframes wave {
        0%, 100% { transform: scaleY(0.4); }
        50% { transform: scaleY(1); }
    }

    /* Status badge */
    .badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        border: 1px solid var(--border);
        color: var(--muted);
    }
    .badge.completed { color: var(--accent); border-color: var(--accent); }
    .badge.failed { color: #F87171; border-color: #F87171; }
    .badge.processing { color: var(--amber); border-color: var(--amber); }

    /* Sidebar select boxes — fix low-contrast text/background */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: var(--text) !important;
    }
    /* Dropdown popover menus render outside the sidebar in a portal */
    div[data-baseweb="popover"] [role="listbox"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    div[data-baseweb="popover"] [role="option"] {
        color: var(--text) !important;
        background-color: var(--surface) !important;
    }
    div[data-baseweb="popover"] [role="option"]:hover {
        background-color: var(--accent-soft) !important;
    }

    /* File uploader — fix low-contrast text/buttons */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg) !important;
        border: 1px dashed var(--border) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: var(--muted) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* Hide the "Press Enter to apply" hint on text inputs */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Bordered containers (st.container(border=True)) styled as cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* Chat bubbles, WhatsApp-style */
    .chat-row {
        display: flex;
        margin-bottom: 0.6rem;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }

    .chat-bubble {
        max-width: 75%;
        padding: 0.6rem 1rem;
        border-radius: 14px;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .chat-bubble.user {
        background-color: var(--accent);
        color: #0B0E14;
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.assistant {
        background-color: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-bottom-left-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:1rem;">
        <div class="waveform">
            <span></span><span></span><span></span><span></span><span></span><span></span>
        </div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700;">
                Verbatim
            </div>
            <div style="color:var(--muted); font-size:0.9rem;">
                Turn any video, meeting, or upload into a transcript, summary, and a chat-ready knowledge base.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow">New job</div>', unsafe_allow_html=True)
    st.markdown("### Process a source")

    source_kind = st.radio("Source", ["YouTube URL", "Upload file"], label_visibility="collapsed")

    youtube_url = None
    uploaded_file = None

    if source_kind == "YouTube URL":
        youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm"],
        )

    content_type = st.selectbox("Content type", ["meeting", "youtube"])
    language = st.selectbox("Language", ["english", "hinglish", "benglish", "bengali", "hindi"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("Process", use_container_width=True)

    if st.session_state.job_id:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Current job</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mono" style="max-height:60px;">{st.session_state.job_id}</div>', unsafe_allow_html=True)
        if st.button("Start a new job", use_container_width=True):
            try:
                requests.delete(f"{BACKEND_URL}/job/{st.session_state.job_id}")
            except requests.exceptions.ConnectionError:
                pass
            st.session_state.job_id = None
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# SUBMIT HANDLER
# ---------------------------------------------------------------------------

if submit:
    st.session_state.result = None
    st.session_state.chat_history = []

    try:
        if source_kind == "YouTube URL":
            if not youtube_url:
                st.error("Enter a YouTube URL first.")
                st.stop()

            resp = requests.post(
                f"{BACKEND_URL}/process/youtube",
                json={"url": youtube_url, "content_type": content_type, "language": language},
            )
        else:
            if uploaded_file is None:
                st.error("Upload a file first.")
                st.stop()

            resp = requests.post(
                f"{BACKEND_URL}/process/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                data={"content_type": content_type, "language": language},
            )

        if resp.status_code not in (200, 202):
            st.error(f"Backend rejected the request: {resp.text}")
            st.stop()

        job_id = resp.json()["job_id"]
        st.session_state.job_id = job_id

    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach the backend at {BACKEND_URL}. Is it running?")
        st.stop()

# ---------------------------------------------------------------------------
# POLL FOR JOB STATUS
# ---------------------------------------------------------------------------

if st.session_state.job_id and st.session_state.result is None:
    job_id = st.session_state.job_id

    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <div class="eyebrow">Working</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700;">
                Processing your source
            </div>
            <div style="color:var(--muted); font-size:0.9rem; margin-top:0.25rem;">
                This can take a few minutes for longer videos.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()

    while True:
        try:
            status_resp = requests.get(f"{BACKEND_URL}/status/{job_id}")
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the backend at {BACKEND_URL}.")
            st.stop()

        if status_resp.status_code != 200:
            st.error("This job no longer exists on the backend.")
            st.session_state.job_id = None
            st.stop()

        job_status = status_resp.json()["status"]

        if job_status == "completed":
            result_resp = requests.get(f"{BACKEND_URL}/result/{job_id}")
            st.session_state.result = result_resp.json()
            break

        if job_status == "failed":
            error_msg = status_resp.json().get("error", "Unknown error")
            st.error(f"Processing failed: {error_msg}")
            st.session_state.job_id = None
            st.stop()

        status_placeholder.markdown(
            f'<div class="mono" style="text-align:center; max-height:none;">Current step: <strong>{job_status}</strong></div>',
            unsafe_allow_html=True,
        )
        time.sleep(3)

    st.rerun()

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

if st.session_state.result:
    result = st.session_state.result
    insights = result["insights"]
    is_youtube = result["content_type"] == "youtube"

    st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
    st.markdown(f"## {result['title']}")

    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Insights", "Transcript", "Chat"]
    )

    with tab_summary:
        with st.container(border=True):
            st.markdown(result["summary"])

    with tab_insights:
        if is_youtube:
            st.markdown('<div class="eyebrow">Key topics</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["key_topics"])

            st.markdown('<div class="eyebrow">Takeaways</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["takeaways"])

            st.markdown('<div class="eyebrow">Notable quotes</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["notable_quotes"])
        else:
            st.markdown('<div class="eyebrow">Action items</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["action_items"])

            st.markdown('<div class="eyebrow">Key decisions</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["key_decisions"])

            st.markdown('<div class="eyebrow">Open questions</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(insights["open_questions"])

    with tab_transcript:
        st.markdown(f'<div class="mono">{result["transcript"]}</div>', unsafe_allow_html=True)

    with tab_chat:
        for entry in st.session_state.chat_history:
            role = entry["role"]
            st.markdown(
                f"""
                <div class="chat-row {role}">
                    <div class="chat-bubble {role}">{entry["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        question = st.chat_input("Ask something about this transcript...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            try:
                chat_resp = requests.post(
                    f"{BACKEND_URL}/chat/{st.session_state.job_id}",
                    json={"question": question},
                )
                if chat_resp.status_code == 200:
                    answer = chat_resp.json()["answer"]
                else:
                    answer = f"Error: {chat_resp.text}"
            except requests.exceptions.ConnectionError:
                answer = f"Can't reach the backend at {BACKEND_URL}."

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

elif not st.session_state.job_id:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 3rem;">
            <div style="color:var(--muted);">
                Paste a YouTube link or upload a file in the sidebar, then hit
                <span style="color:var(--accent);">Process</span> to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )