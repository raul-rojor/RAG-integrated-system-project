"""
Command-line runner for the Music Recommender.

Usage:
  python -m src.main "late-night drive after a rough day"

Describe a situation, mood, or vibe in plain language. With ANTHROPIC_API_KEY set the app runs
the full Claude + RAG pipeline (src/rag.py); otherwise it falls back to the deterministic offline
path over the local catalog. All ranking is done by the unchanged scorer in src/recommender.py.
"""

import sys
import textwrap
from pathlib import Path

# Put the project root and this src/ directory on sys.path so imports work whether launched as
# `python -m src.main`, as a plain script, or from any working directory.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.rag import build_backend, load_codebook, recommend
from src.recommender import load_songs

WIDTH = 62
TITLE_WIDTH = 40
DEFAULT_QUERY = "a rainy afternoon studying, lo-fi but a little jazzy"


def _score_bar(score: float, width: int = 10) -> str:
    """A small proportional bar, e.g. ███████░░░ for a score of ~0.7."""
    filled = round(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)


def _truncate(text: str, limit: int) -> str:
    """Trim text to `limit` characters with an ellipsis when it overflows."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(query: str, recs, online: bool, notes=None, model=None) -> str:
    """Build a clean, ranked terminal layout: the query, the mode/model, notices, and each pick."""
    mode = "AI (Claude + RAG)" if online else "offline (deterministic)"
    lines = ["=" * WIDTH, "  🎵  Top Music Recommendations",
             f'  for  "{_truncate(query, WIDTH - 10)}"', f"  mode: {mode}"]
    if online and model:
        lines.append(f"  model: {model} (fixed)")
    for note in notes or []:
        wrapped = textwrap.wrap(note, WIDTH - 4) or [note]
        lines.append(f"  ⚠ {wrapped[0]}")
        lines.extend(f"    {continuation}" for continuation in wrapped[1:])
    lines.append("=" * WIDTH)
    if not recs:
        return "\n".join(lines + ["", "  No matching songs found.", "", "=" * WIDTH])
    for rank, (song, score, expl) in enumerate(recs, start=1):
        name = _truncate(f"{song['title']} — {song['artist']}", TITLE_WIDTH)
        lines += ["", f"  {rank}. {name:<{TITLE_WIDTH}}  {_score_bar(score)} {score:.2f}"]
        parts = expl.split("; ") if "; " in expl else [expl]
        for i, part in enumerate(parts):
            lines.append(f"        {'└─' if i == len(parts) - 1 else '├─'} {part}")
        if song.get("source"):
            lines.append(f"           ↪ source: {_truncate(str(song['source']), TITLE_WIDTH)}")
    return "\n".join(lines + ["", "=" * WIDTH])


def main() -> None:
    """Read the situation from argv (or a default), run the pipeline, and print the result."""
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    catalog = load_songs(str(PROJECT_ROOT / "data" / "songs.csv"))
    codebook = load_codebook(str(PROJECT_ROOT / "knowledge" / "taste_codebook.md"))
    backend = build_backend()
    notes: list = []
    if backend is not None:
        print("  … AI mode: contacting Claude (web search can take ~30-90s)", file=sys.stderr, flush=True)

    def status(message: str) -> None:
        """Print a live progress line to stderr so AI mode never looks frozen."""
        print(f"  … {message}", file=sys.stderr, flush=True)

    recs = recommend(query, backend=backend, catalog=catalog, codebook=codebook, k=5, notes=notes,
                     progress=status if backend is not None else None)
    print(render(query, recs, online=backend is not None, notes=notes, model=getattr(backend, "MODEL", None)))


if __name__ == "__main__":
    main()
