"""
RAG layer: natural-language taste -> grounded music recommendations.

Pipeline (each AI step degrades to a deterministic offline path):
  1. parse_taste        NL situation -> user_prefs, grounded in the local Taste Codebook (RAG)
  2. propose_candidates user_prefs   -> real songs + features, grounded in web docs (RAG)
  3. explain            per-song prose grounded in the scorer's own numeric reasons
All ranking is done by the UNCHANGED deterministic engine in recommender.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from src.recommender import TEMPO_MAX, TEMPO_MIN, score_song

LIST_FEATURES = ("genre", "mood")
NUMERIC_FEATURES = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")
REQUIRED_CANDIDATE_FIELDS = ("title", "artist") + LIST_FEATURES + NUMERIC_FEATURES

_TOKEN = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Retrieval — the "R" in RAG: read the corpus BEFORE the model generates.
# ---------------------------------------------------------------------------
def load_codebook(path: str = "knowledge/taste_codebook.md") -> str:
    """Return the Taste Codebook text, or '' if the corpus file is absent."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _tokens(text: str) -> set:
    """Lowercase alphanumeric token set for lexical overlap scoring."""
    return set(_TOKEN.findall(text.lower()))


def retrieve_sections(query: str, codebook: str, k: int = 4) -> str:
    """Return the k codebook sections most lexically relevant to the query (RAG retrieval)."""
    sections = [s.strip() for s in re.split(r"\n(?=#+ )", codebook or "") if s.strip()]
    q = _tokens(query)
    hits = [s for s in sections if q & _tokens(s)]
    hits.sort(key=lambda s: len(q & _tokens(s)), reverse=True)
    return "\n\n".join(hits[:k])


# ---------------------------------------------------------------------------
# Offline parser — a compact lexicon mirroring the codebook (used with no API key).
# ---------------------------------------------------------------------------
_LEXICON = [
    ({"study", "studying", "focus", "lock in", "homework", "concentrate"},
     {"genre": ["lofi", "ambient"], "mood": ["focused", "chill"], "energy": 0.37, "tempo_bpm": 80, "acousticness": 0.70, "valence": 0.50}),
    ({"sleep", "sleeping", "bed", "wind down", "relax", "calm", "unwind", "chill"},
     {"genre": ["ambient", "lofi"], "mood": ["calm", "chill"], "energy": 0.30, "tempo_bpm": 70, "acousticness": 0.80, "valence": 0.50}),
    ({"gym", "workout", "lift", "lifting", "exercise", "pump", "hype", "training"},
     {"genre": ["pop", "edm", "hip-hop"], "mood": ["intense", "energetic"], "energy": 0.92, "tempo_bpm": 140, "danceability": 0.85, "acousticness": 0.05, "valence": 0.65}),
    ({"run", "running", "jog", "cardio"},
     {"genre": ["edm", "pop"], "mood": ["energetic"], "energy": 0.88, "tempo_bpm": 165, "danceability": 0.70, "acousticness": 0.10, "valence": 0.65}),
    ({"party", "dance", "dancing", "club", "rave"},
     {"genre": ["pop", "disco", "edm"], "mood": ["playful", "euphoric"], "energy": 0.85, "tempo_bpm": 124, "danceability": 0.90, "acousticness": 0.10, "valence": 0.80}),
    ({"drive", "driving", "road", "highway", "car", "commute"},
     {"genre": ["indie", "synthwave", "pop"], "mood": ["moody"], "energy": 0.62, "tempo_bpm": 112, "danceability": 0.68, "acousticness": 0.30, "valence": 0.55}),
    ({"heartbreak", "breakup", "crying", "sad", "lonely", "alone"},
     {"genre": ["blues", "indie", "r&b"], "mood": ["sad", "melancholic"], "energy": 0.35, "tempo_bpm": 80, "danceability": 0.45, "acousticness": 0.60, "valence": 0.25}),
    ({"rage", "angry", "anger", "mad", "catharsis", "scream"},
     {"genre": ["metal", "rock", "punk"], "mood": ["aggressive", "intense"], "energy": 0.95, "tempo_bpm": 160, "acousticness": 0.05, "valence": 0.30}),
    ({"romantic", "date", "love", "intimate"},
     {"genre": ["r&b", "soul", "pop"], "mood": ["romantic"], "energy": 0.50, "tempo_bpm": 85, "danceability": 0.65, "acousticness": 0.40, "valence": 0.75}),
    ({"clean", "cleaning", "chores", "tidy"},
     {"genre": ["pop", "disco"], "mood": ["happy", "playful"], "energy": 0.75, "tempo_bpm": 122, "danceability": 0.85, "acousticness": 0.20, "valence": 0.85}),
    ({"morning", "wake", "sunrise", "coffee"},
     {"genre": ["pop", "indie pop"], "mood": ["happy", "uplifting"], "energy": 0.65, "tempo_bpm": 115, "danceability": 0.70, "valence": 0.78}),
    ({"nostalgia", "nostalgic", "memories", "reminisce"},
     {"genre": ["folk", "indie"], "mood": ["nostalgic", "melancholic"], "energy": 0.50, "tempo_bpm": 100, "valence": 0.50}),
]
_CUES = [
    ({"rain", "rainy"}, {"valence": -0.10, "acousticness": 0.10, "energy": -0.05}),
    ({"sunny", "summer", "bright"}, {"valence": 0.10, "energy": 0.05, "acousticness": -0.10}),
    ({"night", "midnight", "late"}, {"valence": -0.08, "energy": -0.05}),
    ({"autumn", "fall", "winter"}, {"valence": -0.08, "acousticness": 0.10}),
]


def _hits(keywords: set, tokens: set, text: str) -> bool:
    """True if any keyword matches (substring for phrases, whole-token for single words)."""
    return any((kw in text) if " " in kw else (kw in tokens) for kw in keywords)


def _merge_prefs(dicts: List[dict]) -> dict:
    """Union list features and average numeric features across matched lexicon entries."""
    out: Dict[str, Any] = {}
    for key in LIST_FEATURES:
        vals: List[str] = []
        for d in dicts:
            for v in d.get(key, []):
                if v not in vals:
                    vals.append(v)
        if vals:
            out[key] = vals
    for key in NUMERIC_FEATURES:
        nums = [d[key] for d in dicts if key in d]
        if nums:
            out[key] = sum(nums) / len(nums)
    return out


def rule_based_parse(query: str) -> List[dict]:
    """Offline, deterministic NL->prefs via a compact lexicon; returns [{}] when nothing matches."""
    tokens, text = _tokens(query), query.lower()
    matched = [p for kws, p in _LEXICON if _hits(kws, tokens, text)]
    if not matched:
        return [{}]
    prefs = _merge_prefs(matched)
    for kws, nudge in _CUES:
        if _hits(kws, tokens, text):
            for key, delta in nudge.items():
                if key in prefs:
                    prefs[key] += delta
    return [clamp_prefs(prefs)]


# ---------------------------------------------------------------------------
# Guardrails — validate/clamp everything the model (or a user) can get wrong.
# ---------------------------------------------------------------------------
def clamp_prefs(prefs: dict) -> dict:
    """Keep only valid feature keys, coerce genre/mood to lists, clamp numerics to legal ranges."""
    out: Dict[str, Any] = {}
    for key in LIST_FEATURES:
        v = prefs.get(key)
        if v:
            out[key] = [v] if isinstance(v, str) else [str(x) for x in v]
    for key in NUMERIC_FEATURES:
        v = prefs.get(key)
        if v is None:
            continue
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if key == "tempo_bpm":
            out[key] = round(min(TEMPO_MAX, max(TEMPO_MIN, n)), 1)
        else:
            out[key] = round(min(1.0, max(0.0, n)), 3)  # round to hide float-arithmetic noise
    return out


def valid_candidate(c: dict, require_source: bool = False) -> bool:
    """True if a candidate has all required fields, numeric-parseable features, and (if web) a citation."""
    if not all(c.get(f) not in (None, "") for f in REQUIRED_CANDIDATE_FIELDS):
        return False
    if require_source and not c.get("source"):
        return False
    try:
        for k in NUMERIC_FEATURES:
            float(c[k])
    except (TypeError, ValueError):
        return False
    return True


def _clamp_candidate(c: dict) -> dict:
    """Return a copy of a candidate with its numeric features clamped to legal ranges."""
    out = dict(c)
    for k in NUMERIC_FEATURES:
        n = float(c[k])
        out[k] = min(TEMPO_MAX, max(TEMPO_MIN, n)) if k == "tempo_bpm" else min(1.0, max(0.0, n))
    return out


def sanitize_candidates(candidates: Optional[List[dict]], require_source: bool = False) -> List[dict]:
    """Guardrail: drop malformed/unsourced candidates, clamp features, and dedupe by (title, artist)."""
    seen, out = set(), []
    for c in candidates or []:
        if not valid_candidate(c, require_source):
            continue
        key = (str(c["title"]).strip().lower(), str(c["artist"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(_clamp_candidate(c))
    return out


# ---------------------------------------------------------------------------
# Backend abstraction — a Claude adapter in prod; a fake in tests.
# ---------------------------------------------------------------------------
class Backend(Protocol):
    """The three AI operations the pipeline needs; any object with these methods works."""

    def parse(self, query: str, context: str) -> List[dict]: ...
    def propose(self, prefs: List[dict]) -> List[dict]: ...
    def explain(self, song: dict, reasons: List[str]) -> str: ...


# ---------------------------------------------------------------------------
# Pipeline steps.
# ---------------------------------------------------------------------------
def _note(notes: Optional[List[str]], message: str) -> None:
    """Append a human-readable diagnostic to `notes` when the caller is collecting them."""
    if notes is not None:
        notes.append(message)


def _reason(exc: Exception) -> str:
    """Extract a concise message from an exception, unwrapping the API error body when present."""
    text = str(getattr(exc, "message", "") or exc)
    match = re.search(r"['\"]message['\"]:\s*['\"]([^'\"]+)['\"]", text)
    return (match.group(1) if match else text).strip()


def parse_taste(query: str, backend: Optional[Backend] = None, codebook: str = "",
                notes: Optional[List[str]] = None) -> List[dict]:
    """NL situation -> user_prefs; the AI path retrieves codebook sections BEFORE generating."""
    if not query or not query.strip():
        return [{}]
    if backend is None:
        return rule_based_parse(query)
    context = retrieve_sections(query, codebook)  # retrieval happens BEFORE backend.parse()
    try:
        cleaned = [clamp_prefs(p) for p in backend.parse(query, context) if isinstance(p, dict)]
        prefs = [p for p in cleaned if p]
    except Exception as exc:
        _note(notes, f"AI taste parsing failed: {_reason(exc)}; used keyword parser")
        return rule_based_parse(query)
    if prefs:
        return prefs
    _note(notes, "AI taste parsing returned nothing usable; used keyword parser")
    return rule_based_parse(query)


def propose_candidates(prefs: List[dict], backend: Optional[Backend] = None,
                       catalog: Optional[List[dict]] = None,
                       notes: Optional[List[str]] = None) -> List[dict]:
    """user_prefs -> real candidate songs; AI path retrieves web docs first, else uses the local catalog."""
    catalog = catalog or []
    if backend is None:
        return catalog
    try:
        web = sanitize_candidates(backend.propose(prefs), require_source=True)
    except Exception as exc:
        _note(notes, f"web song search failed: {_reason(exc)}; using local catalog")
        return catalog
    if not web:
        _note(notes, "web song search returned no citable songs; using local catalog")
        return catalog
    return web


def explain(song: dict, reasons: List[str], backend: Optional[Backend] = None) -> str:
    """Per-song explanation: grounded Claude prose if available, else the scorer's reasons joined."""
    fallback = "; ".join(reasons) if reasons else "no matching preferences"
    if backend is None:
        return fallback
    try:
        return backend.explain(song, reasons) or fallback
    except Exception:
        return fallback


def score_against_profiles(song: dict, prefs_list: List[dict]):
    """Return the best (score, reasons) for a song across every candidate preference profile."""
    best = (0.0, [])
    for prefs in prefs_list:
        score, reasons = score_song(prefs, song)
        if score >= best[0]:
            best = (score, reasons)
    return best


def recommend(query: str, backend: Optional[Backend] = None, catalog: Optional[List[dict]] = None,
              codebook: str = "", k: int = 5, notes: Optional[List[str]] = None) -> List:
    """Full RAG pipeline: parse -> propose -> guardrail -> deterministic score/rank -> explain."""
    prefs_list = parse_taste(query, backend, codebook, notes=notes)
    candidates = propose_candidates(prefs_list, backend, catalog, notes=notes)
    scored = [(song, *score_against_profiles(song, prefs_list)) for song in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [(song, score, explain(song, reasons, backend)) for song, score, reasons in scored[: max(0, k)]]


# ---------------------------------------------------------------------------
# Claude backend (lazy import; offline unless anthropic + ANTHROPIC_API_KEY present).
# ---------------------------------------------------------------------------
_PROFILE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"profiles": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "genre": {"type": "array", "items": {"type": "string"}},
            "mood": {"type": "array", "items": {"type": "string"}},
            "energy": {"type": "number"}, "tempo_bpm": {"type": "number"},
            "valence": {"type": "number"}, "danceability": {"type": "number"},
            "acousticness": {"type": "number"}},
        "required": ["genre", "mood"]}}},
    "required": ["profiles"]}


def _extract_json(text: str):
    """Return the first JSON array in model text (a fenced ```json block, else the first valid [...] span)."""
    fenced = re.search(r"```json\s*(.+?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()  # raw_decode from each '[' avoids greedy over-matching past the array
    for i, ch in enumerate(text):
        if ch == "[":
            try:
                value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value
    return []


class ClaudeBackend:
    """Anthropic Claude adapter: structured NL parsing, web-grounded candidates, prose explanations."""

    MODEL = "claude-opus-5"

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic()

    def parse(self, query: str, context: str) -> List[dict]:
        """Retrieve-then-generate: the codebook `context` is in the prompt BEFORE prefs are emitted."""
        system = ("Translate a listener's situation into music preferences using ONLY the taste "
                  "codebook below. Output range midpoints; omit any feature the situation doesn't "
                  "determine.\n\n" + (context or ""))
        prompt = (f'Situation: "{query}"\nReturn JSON {{"profiles": [...]}} where each profile may set '
                  "genre[], mood[], energy, tempo_bpm, valence, danceability, acousticness.")
        return self._json(prompt, _PROFILE_SCHEMA, system=system).get("profiles", [])

    def propose(self, prefs: List[dict]) -> List[dict]:
        """Web-grounded generation: search/fetch run BEFORE the model lists real songs + citations."""
        prompt = (
            "Find up to 8 real, existing songs matching these target preferences: "
            f"{json.dumps(prefs)}. Use web_search (and web_fetch on the results) to confirm each song "
            "and gather its audio character. Return ONLY a ```json fenced array; each item {title, "
            "artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, source}. "
            "Numerics 0-1 except tempo_bpm (BPM). Every item MUST include a non-empty `source` URL you "
            "actually retrieved; omit any song you cannot cite.")
        return _extract_json(self._with_web_tools(prompt))

    def explain(self, song: dict, reasons: List[str]) -> str:
        """Rewrite the scorer's numeric reasons as one warm sentence, grounded only in those reasons."""
        prompt = (f"Song: {song.get('title')} by {song.get('artist')}. Scorer reasons: {reasons}. "
                  "Write one warm, friendly sentence (no numbers) grounded only in those reasons.")
        resp = self._client.messages.create(
            model=self.MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
        return self._text(resp)

    def _json(self, prompt: str, schema: dict, system: Optional[str] = None) -> dict:
        """One structured-output call returning schema-validated JSON."""
        kwargs = dict(model=self.MODEL, max_tokens=2048,
                      messages=[{"role": "user", "content": prompt}],
                      output_config={"format": {"type": "json_schema", "schema": schema}})
        if system:
            kwargs["system"] = system
        return json.loads(self._text(self._client.messages.create(**kwargs)))

    def _with_web_tools(self, prompt: str, max_turns: int = 8) -> str:
        """Run one turn with web_search/web_fetch, resuming across pause_turn, and return final text."""
        tools = [{"type": "web_search_20260209", "name": "web_search"},
                 {"type": "web_fetch_20260209", "name": "web_fetch"}]
        messages = [{"role": "user", "content": prompt}]
        resp = None
        for _ in range(max_turns):
            resp = self._client.messages.create(model=self.MODEL, max_tokens=4096, tools=tools, messages=messages)
            if resp.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": resp.content})
        return self._text(resp)

    @staticmethod
    def _text(resp) -> str:
        """Concatenate text blocks, raising on a safety refusal so the pipeline can fall back."""
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError("model refusal")
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def build_backend() -> Optional[Backend]:
    """Return a Claude backend if the SDK and ANTHROPIC_API_KEY are present, else None (offline mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return None
    try:
        return ClaudeBackend()
    except Exception:
        return None
