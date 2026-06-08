# autoscoreNidra

autoscoreNidra is a cross-platform desktop application for automated sleep staging from EEG recordings. It uses modern deep learning and machine learning models to score sleep stages and applies language model sequence correction.

## Architecture

The application is structured into three main layers:

1. **Frontend (`frontend/`)**: A beautiful, modern desktop UI built with **Flutter**, providing dark mode support, fluid micro-animations, channel selections, configuration options, real-time logging, and batch multi-file status tracking.
2. **Bridge (`bridge/`)**: A native **Rust** bridge that manages background task execution, interacts with the operating system, and coordinates the execution of the Python backend engine via subprocesses.
3. **Backend (`backend/`)**: The core sleep scoring engine written in **Python**, leveraging MNE-Python, PyTorch, LightGBM, YASA, and the Greifswald Sleep Stage Classifier (GSSC). It is compiled into a standalone native executable using **PyInstaller** during the build phase.

---

## Directory Structure

```
autoscoreNidra/
├── backend/          # Python scoring engine (existing pipeline core)
│   ├── algorithms.py # SleepStaging adapters (YASA, GSSC, USleep, TinySleepNet, etc.)
│   ├── scorer.py     # Main scoring pipeline and SleepGPT sequence correction
│   ├── cli.py        # Command line interface
│   └── app.py        # Legacy Tkinter GUI (kept for reference / backup)
├── bridge/           # Rust native bridge (crates and coordination logic)
├── frontend/         # Flutter desktop application code
└── .github/          # GitHub Actions CI/CD workflows for compilation
```

---

## Automated Compilation via GitHub Actions

The application is compiled automatically for Windows, macOS, and Linux using a GitHub Actions matrix workflow (`.github/workflows/build.yml`). 

For each operating system:
1. Installs the Flutter SDK, Rust toolchain, and Python 3.12.
2. Bundles the Python backend into a standalone, portable folder structure via PyInstaller.
3. Compiles the Rust native bridge library.
4. Generates the final desktop application (DMG for macOS, MSIX for Windows, AppImage for Linux), embedding the PyInstaller bundle and model weights as assets.
