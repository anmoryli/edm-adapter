"""Generate a full electronic music track with ACE-Step (v2: optimized prompts)."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import load_acestep_model, generate_acestep
from src.audio_io import save_audio, normalize_audio, edm_post_process


STYLE_PROMPTS = {
    "techno": "dark hypnotic techno, 128 BPM, pounding four-on-the-floor kick, deep sub bass, arpeggiated synth lead, industrial warehouse atmosphere, minor key",
    "house": "groovy house music, 124 BPM, punchy kick, funky bass guitar, bright catchy melody, shuffling hi-hats, uplifting major key",
    "trap": "hard-hitting trap, 140 BPM, booming 808 bass, sharp snare, fast hi-hat rolls, dark cinematic melody",
    "ambient": "dreamy ambient, 85 BPM, evolving synth pads, gentle piano, soft drones, lush reverb, no drums",
    "drum_and_bass": "energetic drum and bass, 170 BPM, fast breakbeat, heavy rolling bass, catchy synth hooks",
    "future_bass": "emotional future bass, 150 BPM, bright supersaw stabs, wobbly bass, punchy drums, euphoric drops",
    "trance": "uplifting trance, 138 BPM, soaring synth melody, driving bass, lush string pads, emotional peaks",
    "dubstep": "aggressive dubstep, 140 BPM, massive wobble bass, metallic leads, heavy kicks, intense drops",
}


def main():
    parser = argparse.ArgumentParser(description="生成完整电子音乐 v2（ACE-Step）")
    parser.add_argument("--style", default="techno", choices=list(STYLE_PROMPTS.keys()))
    parser.add_argument("--duration", type=int, default=60, help="时长（秒），最大 240")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--lyrics", default="", help="歌词（留空为纯器乐）")
    args = parser.parse_args()

    prompt = STYLE_PROMPTS[args.style]
    duration = min(args.duration, 240)
    lyrics = args.lyrics if args.lyrics.strip() else "[instrumental]"

    print(f"风格: {args.style} | 时长: {duration}s | 种子: {args.seed}")

    print("\n加载 ACE-Step...")
    pipeline = load_acestep_model(checkpoint_dir="", device="auto", cpu_offload=True, dtype="float32")

    print(f"\n生成中...")
    audio, sr = generate_acestep(pipeline, prompt, lyrics, float(duration), args.seed)
    audio = normalize_audio(audio, peak_db=-1.0)
    audio = edm_post_process(audio, sr)

    if args.output is None:
        args.output = f"outputs/baseline/full_song_v2_{args.style}.wav"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_audio(args.output, audio, sr=sr)

    print(f"\n保存: {args.output} | 时长: {audio.shape[-1]/sr:.1f}s")


if __name__ == "__main__":
    main()
