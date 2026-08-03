"""
Tests for the RAG layer (src/rag.py): one block per pipeline step, plus guardrail edge cases.

Every test runs offline — the Claude backend is replaced by in-memory fakes, so no network or
API key is needed. Fakes let us assert the RAG contract directly: that retrieved document data
reaches the backend BEFORE it produces output.
"""

import pytest

from src.rag import (
    ClaudeBackend,
    _extract_json,
    build_backend,
    clamp_prefs,
    explain,
    parse_taste,
    propose_candidates,
    recommend,
    retrieve_sections,
    rule_based_parse,
    sanitize_candidates,
    score_against_profiles,
    valid_candidate,
)


# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------
def make_candidate(**overrides) -> dict:
    """A neutral, valid web candidate; override any field for a specific case."""
    base = dict(title="T", artist="A", genre="pop", mood="happy", energy=0.5, tempo_bpm=120.0,
                valence=0.5, danceability=0.5, acousticness=0.5, source="http://example.com")
    base.update(overrides)
    return base


class SpyBackend:
    """Records the context passed to parse(); returns canned outputs. Raises where asked."""

    def __init__(self, profiles=None, candidates=None, prose="warm sentence", raise_on=()):
        self.profiles = profiles if profiles is not None else [{"genre": ["pop"]}]
        self.candidates = candidates if candidates is not None else []
        self.prose = prose
        self.raise_on = set(raise_on)
        self.seen_context = None

    def parse(self, query, context):
        if "parse" in self.raise_on:
            raise RuntimeError("boom")
        self.seen_context = context
        return self.profiles

    def propose(self, prefs):
        if "propose" in self.raise_on:
            raise RuntimeError("boom")
        return self.candidates

    def explain(self, song, reasons):
        if "explain" in self.raise_on:
            raise RuntimeError("boom")
        return self.prose


CODEBOOK = (
    "## Deep focus\nstudying focus lofi ambient acousticness 0.7 energy 0.35\n\n"
    "## Party\ndancing disco energy 0.85 danceability 0.9\n"
)


# ---------------------------------------------------------------------------
# Retrieval (the "R" in RAG)
# ---------------------------------------------------------------------------
def test_retrieve_returns_the_relevant_section():
    out = retrieve_sections("studying with focus", CODEBOOK)
    assert "Deep focus" in out and "Party" not in out


def test_retrieve_empty_corpus_is_safe():
    assert retrieve_sections("anything", "") == ""


def test_retrieve_no_overlap_returns_empty():
    assert retrieve_sections("zzzzz qqqqq", CODEBOOK) == ""


# ---------------------------------------------------------------------------
# Offline rule-based parser
# ---------------------------------------------------------------------------
def test_rule_based_parse_maps_known_context():
    prefs = rule_based_parse("time to hit the gym and lift")[0]
    assert "pop" in prefs["genre"] and prefs["energy"] > 0.8


def test_rule_based_parse_handles_multiword_keyword():
    assert rule_based_parse("I need to lock in for finals")[0]["genre"] == ["lofi", "ambient"]


def test_rule_based_parse_applies_weather_cue():
    dry = rule_based_parse("studying")[0]["valence"]
    rainy = rule_based_parse("studying while it is rainy")[0]["valence"]
    assert rainy < dry


def test_rule_based_parse_unknown_returns_empty_profile():
    assert rule_based_parse("qwerty asdf zxcv") == [{}]


# ---------------------------------------------------------------------------
# Guardrail: clamp_prefs
# ---------------------------------------------------------------------------
def test_clamp_prefs_clamps_and_coerces():
    out = clamp_prefs({"genre": "pop", "energy": 5.0, "tempo_bpm": 999, "valence": -3, "bogus": 1})
    assert out["genre"] == ["pop"]           # bare string -> list
    assert out["energy"] == 1.0              # >1 clamped
    assert out["tempo_bpm"] == 220.0         # BPM clamped to bound
    assert out["valence"] == 0.0             # <0 clamped
    assert "bogus" not in out                # unknown key dropped


def test_clamp_prefs_drops_non_numeric():
    assert "energy" not in clamp_prefs({"energy": "loud"})


# ---------------------------------------------------------------------------
# Guardrail: candidate validation
# ---------------------------------------------------------------------------
def test_valid_candidate_requires_all_fields():
    assert valid_candidate(make_candidate())
    assert not valid_candidate(make_candidate(mood=""))


def test_valid_candidate_requires_source_when_asked():
    c = make_candidate(source="")
    assert valid_candidate(c) and not valid_candidate(c, require_source=True)


def test_valid_candidate_rejects_non_numeric_feature():
    assert not valid_candidate(make_candidate(energy="high"))


def test_sanitize_drops_unsourced_clamps_and_dedupes():
    cands = [
        make_candidate(title="Good", energy=9.0),          # kept, energy clamped
        make_candidate(title="NoSrc", source=""),          # dropped (needs source)
        make_candidate(title="Good"),                      # duplicate of first -> dropped
        make_candidate(title="Bad", tempo_bpm="fast"),     # dropped (non-numeric)
    ]
    out = sanitize_candidates(cands, require_source=True)
    assert [c["title"] for c in out] == ["Good"]
    assert out[0]["energy"] == 1.0


def test_sanitize_handles_none():
    assert sanitize_candidates(None) == []


# ---------------------------------------------------------------------------
# parse_taste — offline, AI path, and the RAG "retrieve-before-generate" contract
# ---------------------------------------------------------------------------
def test_parse_taste_empty_query_returns_empty_profile():
    assert parse_taste("   ") == [{}]


def test_parse_taste_offline_uses_rule_based():
    assert parse_taste("gym workout") == rule_based_parse("gym workout")


def test_parse_taste_ai_path_retrieves_codebook_before_generating():
    spy = SpyBackend(profiles=[{"genre": ["lofi"], "energy": 0.35}])
    result = parse_taste("studying and focus", backend=spy, codebook=CODEBOOK)
    # The document data reached the backend BEFORE it returned output:
    assert spy.seen_context and "acousticness" in spy.seen_context
    assert result == [{"genre": ["lofi"], "energy": 0.35}]


def test_parse_taste_clamps_ai_output():
    spy = SpyBackend(profiles=[{"genre": ["pop"], "energy": 9.0, "junk": 1}])
    assert parse_taste("party", backend=spy, codebook=CODEBOOK) == [{"genre": ["pop"], "energy": 1.0}]


def test_parse_taste_falls_back_when_backend_raises():
    spy = SpyBackend(raise_on=["parse"])
    assert parse_taste("gym workout", backend=spy, codebook=CODEBOOK) == rule_based_parse("gym workout")


def test_parse_taste_falls_back_on_empty_ai_output():
    spy = SpyBackend(profiles=[{}])
    assert parse_taste("gym workout", backend=spy, codebook=CODEBOOK) == rule_based_parse("gym workout")


# ---------------------------------------------------------------------------
# propose_candidates
# ---------------------------------------------------------------------------
def test_propose_offline_returns_catalog():
    catalog = [make_candidate(title="Local")]
    assert propose_candidates([{}], backend=None, catalog=catalog) == catalog


def test_propose_ai_returns_sanitized_web_candidates():
    web = [make_candidate(title="Web"), make_candidate(title="NoSrc", source="")]
    spy = SpyBackend(candidates=web)
    out = propose_candidates([{}], backend=spy, catalog=[make_candidate(title="Local")])
    assert [c["title"] for c in out] == ["Web"]   # unsourced dropped, catalog not used


def test_propose_ai_empty_falls_back_to_catalog():
    catalog = [make_candidate(title="Local")]
    assert propose_candidates([{}], backend=SpyBackend(candidates=[]), catalog=catalog) == catalog


def test_propose_ai_raise_falls_back_to_catalog():
    catalog = [make_candidate(title="Local")]
    spy = SpyBackend(raise_on=["propose"])
    assert propose_candidates([{}], backend=spy, catalog=catalog) == catalog


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------
def test_explain_offline_joins_reasons():
    assert explain({}, ["a", "b"], backend=None) == "a; b"


def test_explain_offline_no_reasons():
    assert explain({}, [], backend=None) == "no matching preferences"


def test_explain_ai_returns_prose():
    assert explain({}, ["a"], backend=SpyBackend(prose="lovely")) == "lovely"


def test_explain_ai_raise_falls_back():
    assert explain({}, ["a", "b"], backend=SpyBackend(raise_on=["explain"])) == "a; b"


# ---------------------------------------------------------------------------
# score_against_profiles & recommend (end-to-end)
# ---------------------------------------------------------------------------
def test_score_against_profiles_takes_best():
    song = make_candidate(genre="pop", energy=0.8)
    score, _ = score_against_profiles(song, [{"genre": ["rock"]}, {"genre": ["pop"]}])
    assert score == 1.0


def test_recommend_offline_ranks_and_respects_k():
    catalog = [make_candidate(title="Pop", genre="pop"), make_candidate(title="Rock", genre="rock")]
    recs = recommend("party dancing", catalog=catalog, k=1)
    assert len(recs) == 1 and recs[0][0]["title"] == "Pop"


def test_recommend_k_zero_and_empty_catalog():
    assert recommend("party", catalog=[make_candidate()], k=0) == []
    assert recommend("party", catalog=[], k=5) == []


def test_recommend_unmatched_query_still_returns_songs():
    catalog = [make_candidate(title="A"), make_candidate(title="B")]
    recs = recommend("qwerty zxcv", catalog=catalog, k=5)
    assert len(recs) == 2 and all(score == 0.0 for _, score, _ in recs)


def test_recommend_ai_end_to_end_prefers_web_and_explains():
    web = [make_candidate(title="Web", genre="pop"), make_candidate(title="NoSrc", source="")]
    spy = SpyBackend(profiles=[{"genre": ["pop"]}], candidates=web, prose="because pop")
    recs = recommend("upbeat pop", backend=spy, catalog=[make_candidate(title="Local")], codebook=CODEBOOK, k=5)
    titles = [song["title"] for song, _, _ in recs]
    assert titles == ["Web"] and recs[0][2] == "because pop"   # unsourced dropped, prose used


# ---------------------------------------------------------------------------
# Small utilities & backend gating
# ---------------------------------------------------------------------------
def test_extract_json_fenced_bracket_and_junk():
    assert _extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert _extract_json('noise [{"a": 2}] noise') == [{"a": 2}]
    assert _extract_json("no json here") == []


def test_extract_json_ignores_trailing_citation_brackets():
    # Regression: a greedy [.*] used to span past the array into citation markers like [1].
    assert _extract_json('here [{"a": 3}] per source [1] and [2]') == [{"a": 3}]


# ---------------------------------------------------------------------------
# Observability — visible fallback notices
# ---------------------------------------------------------------------------
def test_propose_notes_on_backend_exception():
    notes = []
    propose_candidates([{}], backend=SpyBackend(raise_on=["propose"]),
                       catalog=[make_candidate()], notes=notes)
    assert notes and "failed" in notes[0]


def test_notice_surfaces_the_exception_reason():
    # The concrete error message (e.g. "credit balance is too low") must reach the user, not just the class.
    class Broke:
        def parse(self, q, c):
            raise RuntimeError("Error code: 400 - {'error': {'message': 'credit balance is too low'}}")
        def propose(self, prefs): return []
        def explain(self, s, r): return ""
    notes = []
    parse_taste("gym", backend=Broke(), codebook=CODEBOOK, notes=notes)
    assert "credit balance is too low" in notes[0]


def test_propose_notes_on_no_citable_songs():
    notes = []
    propose_candidates([{}], backend=SpyBackend(candidates=[make_candidate(source="")]),
                       catalog=[make_candidate()], notes=notes)
    assert notes and "no citable" in notes[0]


def test_propose_success_adds_no_note():
    notes = []
    propose_candidates([{}], backend=SpyBackend(candidates=[make_candidate(title="Web")]),
                       catalog=[make_candidate(title="Local")], notes=notes)
    assert notes == []


def test_parse_taste_notes_on_failure():
    notes = []
    parse_taste("gym", backend=SpyBackend(raise_on=["parse"]), codebook=CODEBOOK, notes=notes)
    assert notes and "parsing failed" in notes[0]


def test_recommend_threads_notes_to_caller():
    notes = []
    recommend("upbeat pop", backend=SpyBackend(raise_on=["propose"]),
              catalog=[make_candidate()], codebook=CODEBOOK, notes=notes, k=3)
    assert any("local catalog" in n for n in notes)


def test_build_backend_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_backend() is None


def test_claude_backend_extract_and_refusal_helpers():
    # _text raises on a refusal so the pipeline can fall back.
    class R:
        stop_reason = "refusal"
        content = []
    with pytest.raises(RuntimeError):
        ClaudeBackend._text(R())
