"""Caption generation for electronic music audio segments."""

import random


def bpm_to_text(bpm: float) -> str:
    if bpm < 90:
        return "slow"
    elif bpm < 115:
        return "mid-tempo"
    elif bpm < 135:
        return "danceable"
    elif bpm < 155:
        return "fast"
    else:
        return "very fast"


def energy_to_text(rms_mean: float) -> str:
    if rms_mean < 0.03:
        return "soft"
    elif rms_mean < 0.08:
        return "moderate energy"
    else:
        return "high energy"


def bass_to_text(low_freq_ratio: float) -> str:
    if low_freq_ratio > 0.45:
        return "deep bass"
    elif low_freq_ratio > 0.25:
        return "solid bass"
    else:
        return "light bass"


def genre_to_mood_options(genre: str) -> list[str]:
    """Return typical mood descriptors for a genre."""
    mood_map = {
        "techno": ["dark", "hypnotic", "driving", "industrial", "underground"],
        "house": ["warm", "groovy", "uplifting", "soulful", "funky"],
        "trap": ["aggressive", "hard-hitting", "dark", "menacing", "heavy"],
        "ambient": ["dreamy", "ethereal", "serene", "atmospheric", "meditative"],
        "drum and bass": ["energetic", "intense", "liquid", "rolling", "frenetic"],
        "dnb": ["energetic", "intense", "liquid", "rolling", "frenetic"],
        "future_bass": ["emotional", "euphoric", "bright", "melodic", "bubbly"],
        "dubstep": ["heavy", "wobbly", "aggressive", "filthy", "gritty"],
        "trance": ["uplifting", "euphoric", "hypnotic", "elevating", "trance-like"],
        "chillout": ["relaxing", "chill", "laid-back", "smooth", "mellow"],
    }
    return mood_map.get(genre, ["electronic", "atmospheric"])


def genre_to_atmosphere(genre: str) -> str:
    """Return typical atmosphere description for a genre."""
    atmo_map = {
        "techno": random.choice(["warehouse", "underground club", "dark basement", "industrial"]),
        "house": random.choice(["beach club", "rooftop party", "warm lounge", "summer festival"]),
        "trap": random.choice(["dark club", "late night studio", "urban night", "street"]),
        "ambient": random.choice(["vast space", "deep ocean", "night sky", "forest"]),
        "drum and bass": random.choice(["rave", "jungle", "warehouse", "festival"]),
        "dnb": random.choice(["rave", "jungle", "warehouse", "festival"]),
        "future_bass": random.choice(["dreamscape", "cloud city", "neon city", "stadium"]),
        "dubstep": random.choice(["underground", "dark venue", "club basement", "arena"]),
        "trance": random.choice(["trance festival", "stadium", "sunrise set", "main stage"]),
        "chillout": random.choice(["sunset beach", "mountain view", "cozy room", "garden"]),
    }
    return atmo_map.get(genre, "club")


def genre_to_instruments(genre: str) -> str:
    """Return typical instrument/sound descriptions for a genre."""
    inst_map = {
        "techno": "heavy kick drum, deep bass, metallic hi-hats, synthesizer textures",
        "house": "four-on-the-floor kick, warm synth chords, groovy bassline, shuffling hi-hats",
        "trap": "deep 808 bass, fast hi-hats, sharp snare, dark synth melodies",
        "ambient": "soft synth pads, deep sub bass, gentle textures, atmospheric reverb",
        "drum and bass": "fast breakbeat drums, rolling bassline, sharp percussion, amen chops",
        "dnb": "fast breakbeat drums, rolling bassline, sharp percussion, amen chops",
        "future_bass": "bright supersaw chords, punchy drums, wobbly synths, vocal chops",
        "dubstep": "wobbly bass, heavy kick, sharp snare, dub sirens",
        "trance": "euphoric synth leads, driving bassline, arpeggiated sequences, pads",
        "chillout": "soft pads, gentle guitar, mellow keys, subtle percussion",
    }
    return inst_map.get(genre, "electronic drums, synth bass, synthesizer pads")


def build_caption(
    bpm: float,
    rms_mean: float,
    low_freq_ratio: float,
    genre: str = "electronic",
    mood: str | None = None,
    clip_type: str = "loop",
) -> str:
    """Build a structured caption from audio features.

    Format: [BPM] BPM [energy] [mood] [genre] [clip type] with [instruments], [atmosphere]
    """
    bpm_val = int(round(bpm))
    energy_desc = energy_to_text(rms_mean)
    bass_desc = bass_to_text(low_freq_ratio)

    if mood is None:
        mood_options = genre_to_mood_options(genre)
        mood_desc = random.choice(mood_options)
    else:
        mood_desc = mood

    instruments = genre_to_instruments(genre)
    atmosphere = genre_to_atmosphere(genre)

    caption = (
        f"{bpm_val} BPM {energy_desc} {mood_desc} {genre} {clip_type} "
        f"with {instruments}, and {atmosphere} atmosphere"
    )
    return caption


def build_caption_from_features(features: dict, genre: str = "electronic", mood: str | None = None) -> str:
    """Build caption from a features dict (as returned by extract_all_features)."""
    return build_caption(
        bpm=features.get("bpm", 120),
        rms_mean=features.get("rms_mean", 0.05),
        low_freq_ratio=features.get("low_freq_ratio", 0.3),
        genre=genre,
        mood=mood,
    )
