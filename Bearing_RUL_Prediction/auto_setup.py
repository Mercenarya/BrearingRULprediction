#!/usr/bin/env python
# auto_setup.py
"""
Automatic setup script for the Bearing RUL Prediction project (Role 2).

This script:
1. Checks Python version.
2. Verifies that PyTorch is installed (installs if missing via pip).
3. Installs any additional required packages (numpy, matplotlib, etc.).
4. Runs a quick sanity check by building the full model and performing a forward pass.
5. Reports success or any errors encountered.

Usage:
    python auto_setup.py
"""

import subprocess
import sys
import importlib
import os

def run_command(cmd, check=True):
    """Run a shell command and return its output."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)
    return result

def check_package(package_name, import_name=None):
    """Check if a Python package is importable."""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"[OK] {package_name} is available.")
        return True
    except ImportError:
        print(f"[MISSING] {package_name} not found.")
        return False

def install_package(package):
    """Install a package using pip."""
    print(f"[INSTALL] Installing {package}...")
    run_command(f"{sys.executable} -m pip install {package}")

def main():
    print("=" * 60)
    print("Bearing RUL Prediction - Auto Setup for Role 2")
    print("=" * 60)

    # 1. Python version check
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[WARNING] Python 3.8+ is recommended.")
    # 2. Ensure pip is up-to-date (optional)
    run_command(f"{sys.executable} -m pip install --upgrade pip", check=False)

    # 3. List of required packages
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]

    for pkg, imp in required:
        if not check_package(pkg, imp):
            # Try to install
            install_package(pkg)
            # Re-check
            if not check_package(pkg, imp):
                print(f"[FATAL] Failed to install {package}. Aborting.")
                sys.exit(1)

    # 4. Verify project structure
    src_path = os.path.join(os.path.dirname(__file__), "src")
    if not os.path.isdir(src_path):
        print(f"[FATAL] Expected src directory not found at {src_path}")
        sys.exit(1)
    print(f"[OK] Source directory found: {src_path}")

    # 5. Quick sanity test: import model components and run demo
    print("\n[INFO] Running sanity check (building model and forward pass)...")
    try:
        # Add project root to sys.path to allow src import
        sys.path.insert(0, os.path.dirname(__file__))
        from src.models.factory import build_model
        from src.models.config import ModelConfig
        import torch

        cfg = ModelConfig(batch_size=2, sequence_length=8, device='cpu')
        model = build_model('full_dual_path', cfg)
        print(f"[OK] Model built: {model.__class__.__name__}")

        batch = {
            "vibration": torch.randn(cfg.batch_size, cfg.sequence_length, 2, 1024),
            "temperature": torch.randn(cfg.batch_size, cfg.sequence_length, 7),
            "temperature_mask": torch.ones(cfg.batch_size, cfg.sequence_length, 7, dtype=torch.bool),
            "rul": torch.rand(cfg.batch_size, 1) * 1000,
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        assert output["rul_pred"].shape == (cfg.batch_size, 1), \
            f"Unexpected output shape: {output['rul_pred'].shape}"
        print(f"[OK] Forward pass successful. Output shape: {output['rul_pred'].shape}")
        # Check for aux if return_aux=True (default False)
        if cfg.return_aux:
            print("[INFO] Auxiliary outputs present:", list(output["aux"].keys()))
        else:
            print("[INFO] return_aux=False (no auxiliary outputs).")
    except Exception as e:
        print(f"[FATAL] Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Setup completed successfully! 🎉")
    print("You can now run demos, tests, or training scripts.")
    print("=" * 60)

if __name__ == "__main__":
    main()