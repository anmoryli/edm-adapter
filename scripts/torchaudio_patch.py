"""torchaudio compatibility patch for Windows (avoids torchcodec dependency)

Import this module before using torchaudio.save/load to bypass the
torchcodec/FFmpeg DLL requirement on Windows.
"""

import sys


def patch_torchaudio():
    """Replace torchaudio.save and torchaudio.load with soundfile-based implementations."""
    try:
        import torchaudio
        import soundfile as sf
        import torch
        import numpy as np
    except ImportError as e:
        print(f"[torchaudio_patch] Warning: could not import dependencies: {e}")
        return

    _orig_save = torchaudio.save
    _orig_load = torchaudio.load

    def _save_with_soundfile(filepath, src, sample_rate, **kwargs):
        wav_np = src.float().cpu().numpy()
        if wav_np.ndim == 1:
            wav_np = wav_np.reshape(1, -1)
        # soundfile expects (samples, channels)
        sf.write(filepath, wav_np.T, sample_rate)

    def _load_with_soundfile(filepath, **kwargs):
        wav, sr = sf.read(filepath, dtype='float32')
        if wav.ndim == 1:
            wav = wav.reshape(1, -1)
        else:
            wav = wav.T  # (samples, channels) -> (channels, samples)
        return torch.from_numpy(wav), sr

    # Check if torchaudio.save actually works; only patch if it fails
    try:
        import tempfile, os
        _test_path = os.path.join(tempfile.gettempdir(), "_torchaudio_test.wav")
        _test_tensor = torch.zeros(1, 1000)
        torchaudio.save(_test_path, _test_tensor, 22050)
        os.remove(_test_path)
        # torchaudio.save works, no need to patch
        return
    except Exception:
        pass

    # Patch
    torchaudio.save = _save_with_soundfile
    torchaudio.load = _load_with_soundfile
    print("[torchaudio_patch] Patched torchaudio.save/load to use soundfile")


# Auto-patch on import
patch_torchaudio()
