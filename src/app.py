"""
Streamlit UI wrapping the recommender.

Run with:  streamlit run src/app.py

Thin front end over src/rag.recommend(): a text box for the natural-language situation, live progress,
the fixed model + mode, any fallback notices, and ranked picks with sources. No core logic lives here.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

from src.rag import MODEL, build_backend, load_codebook, recommend
from src.recommender import load_songs


@st.cache_data
def _catalog() -> list:
    """Load and cache the local song catalog once per session."""
    return load_songs(str(PROJECT_ROOT / "data" / "songs.csv"))


@st.cache_data
def _codebook() -> str:
    """Load and cache the Taste Codebook retrieval corpus once per session."""
    return load_codebook(str(PROJECT_ROOT / "knowledge" / "taste_codebook.md"))


st.set_page_config(page_title="Music Matcher", page_icon="🎵")
st.title("🎵 Music Matcher")
st.caption("Describe a situation, mood, or vibe — get songs that fit.")

mode = st.radio("Mode", ["Offline (free · local catalog)", "Claude (bring your own API key)"], index=0)

backend = None
if mode.startswith("Claude"):
    api_key = st.text_input("Anthropic API key", type="password",
                            help="Used only for this session and never stored. Each run spends your own credits.")
    if api_key.strip():
        backend = build_backend(api_key=api_key.strip())
        if backend is None:
            st.error("Couldn't start Claude mode — is the `anthropic` package installed? Try `pip install anthropic`.")
        else:
            st.info(f"Claude mode · model `{MODEL}` (fixed)")
    else:
        st.warning("Enter your Anthropic API key above to use Claude mode.")
else:
    st.caption("Offline mode — deterministic ranking over the local catalog. No key needed.")

query = st.text_input("Your situation or vibe", placeholder="late-night drive after a rough day")
k = st.slider("How many songs", min_value=1, max_value=10, value=5)

if st.button("Recommend", type="primary") and query.strip():
    if mode.startswith("Claude") and backend is None:
        st.error("Enter a valid Anthropic API key first, or switch to Offline mode.")
        st.stop()
    notes: list = []
    with st.status("Working…", expanded=True) as status:
        recs = recommend(query, backend=backend, catalog=_catalog(), codebook=_codebook(),
                         k=k, notes=notes, progress=lambda msg: status.write(f"… {msg}"))
        status.update(label="Done", state="complete")

    for note in notes:
        st.warning(f"⚠ {note}")

    if backend is not None and getattr(backend, "debug_log", None):
        with st.expander("🔧 Diagnostics — what Claude returned"):
            st.code("\n\n".join(backend.debug_log) or "(no diagnostics captured)")

    if not recs:
        st.error("No matching songs found.")
    for rank, (song, score, expl) in enumerate(recs, start=1):
        st.markdown(f"**{rank}. {song['title']} — {song['artist']}**  ·  score `{score:.2f}`")
        st.progress(min(1.0, max(0.0, score)))
        for part in (expl.split("; ") if "; " in expl else [expl]):
            st.markdown(f"- {part}")
        if song.get("source"):
            st.markdown(f"↪ [source]({song['source']})")
        st.divider()
