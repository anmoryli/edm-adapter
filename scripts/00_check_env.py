"""Check environment: required packages, CUDA, disk space."""

import sys
import shutil


def check_package(name, import_name=None):
    try:
        mod = __import__(import_name or name)
        ver = getattr(mod, "__version__", "installed")
        return True, ver
    except ImportError:
        return False, None


def main():
    print("=" * 60)
    print("EDM-Adapter Environment Check")
    print("=" * 60)

    # Python version
    print(f"\nPython: {sys.version}")

    # Required packages
    packages = [
        ("torch", None),
        ("torchaudio", None),
        ("librosa", None),
        ("soundfile", None),
        ("numpy", None),
        ("pandas", None),
        ("scipy", None),
        ("matplotlib", None),
        ("tqdm", None),
        ("yaml", "yaml"),
        ("gradio", None),
        ("sklearn", "sklearn"),
    ]

    print("\n--- Required Packages ---")
    all_ok = True
    for pkg_name, imp_name in packages:
        ok, ver = check_package(pkg_name, imp_name)
        status = f"OK ({ver})" if ok else "MISSING"
        if not ok:
            all_ok = False
        print(f"  {pkg_name:20s} {status}")

    # Optional packages
    optional = [
        ("stable_audio_tools", None),
        ("audiocraft", None),
        ("essentia", None),
        ("accelerate", None),
        ("transformers", None),
    ]

    print("\n--- Optional Packages ---")
    for pkg_name, imp_name in optional:
        ok, ver = check_package(pkg_name, imp_name)
        status = f"OK ({ver})" if ok else "not installed"
        print(f"  {pkg_name:20s} {status}")

    # CUDA
    print("\n--- GPU ---")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            print("  CUDA available: No (CPU only)")
    except Exception:
        print("  Cannot check CUDA")

    # Disk space
    print("\n--- Disk Space ---")
    total, used, free = shutil.disk_usage(".")
    print(f"  Free: {free / 1e9:.1f} GB")

    print("\n" + "=" * 60)
    if all_ok:
        print("All required packages installed. Ready to go!")
    else:
        print("Some packages are missing. Install with:")
        print("  pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
