"""Prompt templates for the Gradio demo and generation."""

# Style options
STYLES = {
    "techno": {
        "label": "Techno",
        "bpm_range": (120, 140),
        "default_bpm": 128,
        "elements": "heavy kick drum, deep bass, metallic hi-hats, synthesizer textures",
    },
    "house": {
        "label": "House",
        "bpm_range": (118, 132),
        "default_bpm": 124,
        "elements": "four-on-the-floor kick, warm synth chords, groovy bassline, shuffling hi-hats",
    },
    "trap": {
        "label": "Trap",
        "bpm_range": (130, 160),
        "default_bpm": 140,
        "elements": "deep 808 bass, fast hi-hats, sharp snare, dark synth melodies",
    },
    "ambient": {
        "label": "Ambient",
        "bpm_range": (60, 110),
        "default_bpm": 90,
        "elements": "soft synth pads, deep sub bass, gentle textures, atmospheric reverb",
    },
    "drum_and_bass": {
        "label": "Drum & Bass",
        "bpm_range": (160, 180),
        "default_bpm": 170,
        "elements": "fast breakbeat drums, rolling bassline, sharp percussion",
    },
    "future_bass": {
        "label": "Future Bass",
        "bpm_range": (130, 160),
        "default_bpm": 150,
        "elements": "bright supersaw chords, punchy drums, wobbly synths, vocal chops",
    },
}

MOODS = ["dark", "energetic", "dreamy", "aggressive", "relaxing", "hypnotic", "euphoric", "warm"]

CLIP_TYPES = ["loop", "beat", "texture", "drop", "riff"]

DURATIONS = [10, 20, 30]


def build_prompt(style: str, bpm: int, mood: str, clip_type: str = "loop", duration: int = 10) -> str:
    """Build a generation prompt from parameters."""
    style_info = STYLES.get(style, STYLES["techno"])
    elements = style_info["elements"]

    prompt = (
        f"{bpm} BPM {mood} {style.replace('_', ' ')} {clip_type} "
        f"with {elements}"
    )
    return prompt


# Fixed evaluation prompts
EVAL_PROMPTS = [
    {
        "id": "techno_dark",
        "text": "128 BPM dark techno loop with heavy four-on-the-floor kick, rumbling bass and metallic hi-hats",
        "duration": 10,
        "target_bpm": 128,
        "genre": "techno",
    },
    {
        "id": "house_warm",
        "text": "124 BPM energetic house loop with warm chords, groovy bassline and club atmosphere",
        "duration": 10,
        "target_bpm": 124,
        "genre": "house",
    },
    {
        "id": "trap_808",
        "text": "140 BPM trap beat with deep 808 bass, fast hi-hats and sharp snare",
        "duration": 10,
        "target_bpm": 140,
        "genre": "trap",
    },
    {
        "id": "dnb_fast",
        "text": "170 BPM drum and bass loop with fast breakbeat drums and rolling bass",
        "duration": 10,
        "target_bpm": 170,
        "genre": "drum_and_bass",
    },
    {
        "id": "ambient_pad",
        "text": "90 BPM ambient electronic texture with soft synth pads and deep sub bass",
        "duration": 10,
        "target_bpm": 90,
        "genre": "ambient",
    },
    {
        "id": "future_bass",
        "text": "future bass drop with bright supersaw chords, punchy drums and emotional atmosphere",
        "duration": 10,
        "target_bpm": 145,
        "genre": "future_bass",
    },
]
