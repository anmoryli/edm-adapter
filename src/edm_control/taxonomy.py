"""Fixed vocabularies used by the EDM control LoRA pipeline."""

DEFAULT_SECTIONS = [
    "intro",
    "build-up",
    "drop",
    "breakdown",
    "bridge",
    "chorus",
    "outro",
    "loop",
    "unknown",
]

DEFAULT_ENERGIES = ["low", "medium", "high", "very_high"]

DEFAULT_SUBGENRES = [
    "progressive house",
    "melodic house",
    "festival EDM",
    "future bass",
    "electro house",
    "tropical house",
    "piano house",
    "folk EDM",
    "big room house",
    "deep house",
    "dance pop",
]

ENERGY_TO_VALUE = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.78,
    "very_high": 1.00,
}
