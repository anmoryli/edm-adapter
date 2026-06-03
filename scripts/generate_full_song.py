"""Generate a full-length electronic music track using ACE-Step.

ACE-Step can generate up to 4 minutes in a single pass.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import load_acestep_model, generate_acestep
from src.audio_io import save_audio, normalize_audio, edm_post_process


STYLE_PROMPTS = {
    "techno": "dark techno, industrial, hypnotic, 128 BPM, pounding kick drum, deep sub bass, arpeggiated synth, warehouse atmosphere",
    "house": "house music, groovy, warm, 124 BPM, punchy kick, funky bass, bright synth melodies, shuffling hi-hats",
    "trap": "trap, aggressive, dark, 140 BPM, booming 808 bass, sharp snare, rapid hi-hat rolls, cinematic",
    "ambient": "ambient, atmospheric, dreamy, 85 BPM, evolving synth pads, gentle piano, lush reverb, no percussion",
    "drum_and_bass": "drum and bass, energetic, 170 BPM, fast breakbeat, heavy rolling bass, catchy synth hooks",
    "future_bass": "future bass, emotional, euphoric, 150 BPM, supersaw chords, wobbly bass, punchy drums",
    "trance": "trance, uplifting, euphoric, 138 BPM, soaring synth melodies, driving bass, lush pads",
    "dubstep": "dubstep, heavy bass, aggressive, 140 BPM, wobble bass growls, metallic leads, intense drops",
}


def main():
    parser = argparse.ArgumentParser(description="生成完整电子音乐（ACE-Step）")
    parser.add_argument("--style", default="techno", choices=list(STYLE_PROMPTS.keys()))
    parser.add_argument("--duration", type=int, default=60, help="目标时长（秒），最大 240")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="输出路径")
    parser.add_argument("--lyrics", default="", help="歌词（留空为纯器乐）")
    args = parser.parse_args()

    prompt = STYLE_PROMPTS[args.style]
    duration = min(args.duration, 240)
    lyrics = args.lyrics if args.lyrics.strip() else "[instrumental]"

    print(f"风格: {args.style}")
    print(f"时长: {duration}s")
    print(f"提示词: {prompt}")

    # 加载模型
    print("\n加载 ACE-Step...")
    pipeline = load_acestep_model(
        checkpoint_dir="",
        device="auto",
        cpu_offload=True,
        dtype="float32",
    )

    # 生成
    print(f"\n生成中...")
    audio, sr = generate_acestep(
        pipeline=pipeline,
        prompt=prompt,
        lyrics=lyrics,
        duration=float(duration),
        seed=args.seed,
    )

    # 后处理
    audio = normalize_audio(audio, peak_db=-1.0)
    audio = edm_post_process(audio, sr)

    # 保存
    if args.output is None:
        args.output = f"outputs/baseline/full_song_{args.style}.wav"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_audio(args.output, audio, sr=sr)

    duration_actual = audio.shape[-1] / sr
    print(f"\n保存: {args.output}")
    print(f"时长: {duration_actual:.1f}s")
    print(f"采样率: {sr} Hz")


if __name__ == "__main__":
    main()
