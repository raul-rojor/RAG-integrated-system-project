# 🎚️ Taste Codebook — NL situations → musical preferences

> **What this is.** A curated, human-owned **retrieval corpus** for the parse step of the RAG
> recommender. When a user describes a *situation, setting, or vibe* in natural language, Claude
> retrieves the relevant rows below and grounds its translation into a `user_prefs` dict for the
> deterministic scorer in [`src/recommender.py`](../src/recommender.py).
>
> **Why a codebook, not a lookup.** The space of natural-language situations is effectively
> infinite. This document does **not** try to enumerate situations. It encodes the small number of
> *dimensions* situations reduce to — emotional target (valence/arousal), functional context,
> setting cues, and vibe descriptors — each mapped to the seven features the scorer uses:
> `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`.
>
> **Provenance & bias.** Distilled from Russell's circumplex model of affect (valence–arousal),
> Thayer's energy–stress model, Spotify audio-feature definitions, and the AllMusic mood/theme
> vocabulary. These mappings are **culturally situated priors, not universal truths** — "sad" or
> "party" music varies by person and culture. This file is the *editable, reviewable* home for that
> bias: designers own it, version it, and correct it. Values are **ranges (priors), not hard rules.**

---

## How the model should use this corpus

1. Read the user's situation; identify its **emotional target** (§1) plus any **functional
   context** (§2), **setting/time cues** (§3), and **vibe descriptors** (§4).
2. Combine them: functional context sets the base feature signature; setting cues and descriptors
   *nudge* it; the valence–arousal target sanity-checks `valence` and `energy`.
3. Emit a `user_prefs` dict. Use **midpoints** of the retrieved ranges as targets.
4. **Omit any feature the situation doesn't determine** — the scorer renormalizes over the features
   you provide (`_score_core`), so silence is safer than a guess.
5. If the situation is genuinely multi-vibe ("hype but also emotional"), emit **multiple profiles**.
6. Every field should be traceable to a row below — cite the row(s) used so a human can audit it.

---

## §1 — Emotional target (valence–arousal quadrants)

The backbone. Every situation lands somewhere on these two axes.

| Quadrant (arousal × valence) | `energy` | `valence` | Typical `mood` | Feels like |
|---|---|---|---|---|
| High energy · High valence | 0.75–1.0 | 0.7–0.95 | happy, euphoric, playful | party, triumphant, celebratory |
| High energy · Low valence | 0.8–1.0 | 0.15–0.4 | aggressive, intense, tense | rage, catharsis, anxiety |
| Low energy · High valence | 0.2–0.5 | 0.6–0.85 | calm, content, serene, romantic | cozy, tender, peaceful |
| Low energy · Low valence | 0.15–0.45 | 0.1–0.4 | sad, melancholic, somber | heartbreak, grief, rainy blues |

---

## §2 — Functional / activity context

Sets the base feature signature. Ranges are 0–1 except `tempo_bpm` (raw BPM).

| Context | energy | tempo_bpm | danceability | acousticness | valence | Typical genres | Typical moods |
|---|---|---|---|---|---|---|---|
| Deep focus / studying | 0.30–0.45 | 70–90 | 0.40–0.60 | 0.50–0.85 | 0.40–0.60 | lofi, ambient, classical, instrumental | focused, chill |
| Reading / calm | 0.25–0.45 | 65–90 | 0.35–0.55 | 0.55–0.90 | 0.45–0.65 | classical, ambient, folk | calm, relaxed |
| Sleep / winding down | 0.10–0.30 | 50–70 | 0.20–0.40 | 0.70–0.95 | 0.30–0.55 | ambient, lofi, classical | calm, chill |
| Meditation / yoga | 0.10–0.30 | 50–75 | 0.25–0.45 | 0.70–0.95 | 0.50–0.70 | ambient, new age | calm, serene |
| Wake up / morning boost | 0.55–0.75 | 100–125 | 0.60–0.80 | 0.20–0.50 | 0.65–0.85 | pop, indie pop | happy, uplifting |
| Workout / gym | 0.85–1.00 | 125–160 | 0.70–0.95 | 0.00–0.15 | 0.50–0.80 | pop, edm, hip-hop, techno | intense, energetic |
| Running (cadence) | 0.80–0.95 | 150–180 | 0.60–0.85 | 0.00–0.20 | 0.50–0.80 | edm, pop, hip-hop | energetic, intense |
| Party / dancing | 0.75–0.95 | 118–130 | 0.80–0.98 | 0.00–0.20 | 0.70–0.90 | pop, disco, edm, hip-hop | playful, euphoric |
| Pre-game / hype | 0.80–0.95 | 120–140 | 0.75–0.90 | 0.00–0.20 | 0.70–0.90 | hip-hop, pop, phonk | confident, euphoric |
| Chill / unwind | 0.30–0.50 | 80–105 | 0.50–0.70 | 0.30–0.70 | 0.50–0.70 | lofi, r&b, indie | relaxed, chill |
| Cleaning / chores | 0.60–0.80 | 110–128 | 0.70–0.90 | 0.10–0.40 | 0.70–0.90 | pop, disco | happy, playful |
| Cooking / dinner background | 0.40–0.60 | 90–115 | 0.55–0.75 | 0.30–0.60 | 0.60–0.80 | jazz, soul, bossa nova | relaxed, playful |
| Commute / daytime drive | 0.55–0.75 | 100–125 | 0.55–0.75 | 0.20–0.50 | 0.55–0.80 | pop, indie, rock | uplifting, hopeful |
| Night drive (solo, reflective) | 0.50–0.70 | 100–120 | 0.55–0.75 | 0.20–0.40 | 0.35–0.55 | synthwave, indie, r&b | moody |
| Road trip (group) | 0.65–0.85 | 110–135 | 0.60–0.80 | 0.15–0.45 | 0.70–0.90 | rock, pop, country | hopeful, uplifting |
| Gaming | 0.70–0.95 | 120–160 | 0.50–0.80 | 0.00–0.20 | 0.50–0.75 | electronic, techno, metal | intense, euphoric |
| Romantic / date | 0.35–0.60 | 70–100 | 0.50–0.75 | 0.20–0.60 | 0.60–0.85 | r&b, soul, pop | romantic |
| Heartbreak / crying | 0.20–0.45 | 60–90 | 0.30–0.55 | 0.40–0.85 | 0.10–0.35 | blues, singer-songwriter, classical | sad, melancholic |
| Rage / catharsis | 0.85–1.00 | 140–180 | 0.35–0.60 | 0.00–0.10 | 0.15–0.40 | metal, rock, punk | aggressive, intense |
| Nostalgia / reminiscing | 0.40–0.65 | 85–115 | 0.50–0.70 | 0.30–0.70 | 0.40–0.65 | folk, indie, oldies | nostalgic, melancholic |
| Confidence / main-character walk | 0.70–0.90 | 90–120 | 0.75–0.90 | 0.00–0.25 | 0.55–0.80 | hip-hop, pop | confident |
| Background / dinner party | 0.40–0.60 | 95–115 | 0.60–0.75 | 0.30–0.60 | 0.65–0.85 | jazz, soul, disco | relaxed, playful |

---

## §3 — Setting, time & weather cues (nudges)

Applied *on top of* a functional context. `+`/`−` = shift the base value up/down.

| Cue | energy | valence | acousticness | Mood lean |
|---|---|---|---|---|
| Rainy | − | − | + | melancholic, cozy, chill |
| Sunny / summer | + | + | − | happy, uplifting |
| Sunrise / early morning | slight + | + | slight + | hopeful |
| Midnight / late night | − | − | slight − | moody, reflective |
| Autumn | − | − | + | nostalgic, melancholic |
| Winter | − | − | + | calm, melancholic |
| Beach / tropical | + | + | − | uplifting, playful (reggae, tropical) |
| City night | mid | − | − | moody (synthwave, r&b) |

---

## §4 — Vibe & subculture descriptor glossary

Users often name a *vibe* instead of a mood. Map to nearest genre + signature, then let §1–§3 refine.

| Descriptor | Nearest genre(s) | Signature nudge | Mood |
|---|---|---|---|
| dark academia | classical, instrumental, lofi | acousticness +, valence −, energy − | moody, focused |
| lo-fi beats / study beats | lofi | energy low-mid, acousticness + | chill, focused |
| phonk / drift phonk | phonk, electronic, hip-hop | energy +, valence −, danceability + | aggressive, moody |
| yacht rock | soft rock, pop | valence +, energy mid | relaxed, playful |
| shoegaze / dream pop | indie, dream pop | acousticness mid, energy mid | moody, dreamy |
| cottagecore | folk, acoustic | acousticness +, valence + | nostalgic, calm |
| hyperpop | pop, electronic | energy +, danceability + | playful, euphoric |
| sad girl / sad boy | indie, bedroom pop | valence −, energy − | melancholic |
| coquette / dreamy | dream pop, indie pop | valence mid-high, energy low-mid | romantic, dreamy |
| rage / hype (trap) | trap, edm, hip-hop | energy +, tempo + | intense, euphoric |
| yearning / wistful | indie, folk | valence −, acousticness + | nostalgic, melancholic |

Plain adjectives map onto §1 directly: *euphoric / triumphant* → high energy·high valence;
*tense / brooding* → high energy·low valence; *serene / tender* → low energy·high valence;
*somber / wistful* → low energy·low valence.

---

## §5 — Worked examples (few-shot exemplars)

**"Late-night solo drive on the highway after a long, rough day."**
→ Night drive (§2) + midnight (§3) + low-valence target (§1). →
`{"genre": ["synthwave", "indie", "r&b"], "mood": ["moody"], "energy": 0.58, "tempo_bpm": 110, "danceability": 0.65, "acousticness": 0.28, "valence": 0.4}`

**"Cleaning my apartment on a bright Sunday morning."**
→ Cleaning/chores (§2) + sunny/morning (§3) + high-energy·high-valence (§1). →
`{"genre": ["pop", "disco"], "mood": ["happy", "playful"], "energy": 0.75, "tempo_bpm": 122, "danceability": 0.85, "acousticness": 0.2, "valence": 0.85}`

**"Studying for finals, need to lock in, and it's raining outside."**
→ Deep focus (§2) + rainy (§3) + low-arousal·mid-valence (§1). →
`{"genre": ["lofi", "ambient"], "mood": ["focused", "chill"], "energy": 0.35, "tempo_bpm": 78, "danceability": 0.5, "acousticness": 0.78, "valence": 0.5}`

**"Something hype for the gym but I'm also kind of in my feelings today."** *(multi-vibe → two profiles)*
→ Profile A (workout, §2): `{"genre": ["pop", "hip-hop", "edm"], "mood": ["intense"], "energy": 0.92, "tempo_bpm": 140, "danceability": 0.85, "acousticness": 0.05, "valence": 0.6}`
→ Profile B (introspective hip-hop): `{"genre": ["hip-hop", "r&b"], "mood": ["moody", "confident"], "energy": 0.7, "tempo_bpm": 95, "danceability": 0.7, "valence": 0.4}`
