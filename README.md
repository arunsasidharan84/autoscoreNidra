# autoscoreNidra — Python Automated Sleep EEG Scoring Pipeline

**autoscoreNidra** is a standalone Python package and command-line utility for automated 30-second epoch sleep staging from one or more EEG channels. It leverages state-of-the-art deep learning and machine learning models to score sleep stages and applies language model sequence correction (SleepGPT).

---

## 🛠️ Components & Architecture

The repository is structured as a pure Python engine:
```
autoscoreNidra/
├── backend/
│   ├── algorithms.py  # Pretrained model adapters (YASA, U-Sleep, Luna, GSSC, PhysioEx)
│   ├── scorer.py      # Core consensus logic & SleepGPT sequence correction
│   ├── cli.py         # Command-line interface for batch processing
│   ├── app.py         # Tkinter-based desktop interface (backup/reference UI)
│   └── requirements.txt  # Python packages and environment specifications
```

---

## 🔬 Supported Sleep Staging Models

The pipeline integrates 9 machine learning and deep learning base scorers:

1.  **YASA LightGBM Consensus (`yasa`)**: YASA-based LightGBM sleep stager.
2.  **Offline U-Sleep Consensus (`usleep`)**: Re-implementation of the U-Sleep convolutional architecture via Braindecode, using a local pretrained Sleep Physionet checkpoint. Fallback to U-Sleep Web API is supported if `USLEEP_API_TOKEN` is configured.
3.  **Luna POPS (`luna`)**: Pretrained Luna POPS `s2` model resources executing via `lunapi`.
4.  **Greifswald Sleep Stage Classifier (`gssc`)**: The GSSC classifier adapter for clinical datasets.
5.  **TinySleepNet (`tinysleepnet`)**: PhysioEx-based TinySleepNet PyTorch checkpoint.
6.  **SeqSleepNet (`seqsleepnet`)**: PhysioEx-based SeqSleepNet sequence-to-sequence model.
7.  **SleepTransformer (`sleeptransformer`)**: PhysioEx-based transformer model.
8.  **Dreamento (`dreamento`)**: Dreamento-style offline scoring utilizing its LightGBM/YASA-based features.
9.  **SleepEEGpy (`sleepeegpy`)**: MNE-Python and YASA based sleep staging.

---

## 🤖 Sequence Correction (SleepGPT)

**SleepGPT** is a sequence correction module that is applied on top of any selected base scorer. It models the sleep stage sequence as a language generation task, correcting physiologically implausible stage transitions (e.g., transition from N3 directly to REM).

---

## 🚀 Running & Installing Locally

### 1. Set Up the Environment
Create a virtual environment and install dependencies:
```sh
# Create virtual environment
python -m venv sleep_env

# Activate environment (macOS/Linux)
source sleep_env/bin/activate

# Activate environment (Windows)
sleep_env\Scripts\activate
```

### 2. Install Requirements
Install dependencies using standard pip or `uv` manager:
```sh
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

---

## ⌨️ CLI Usage Examples

All command-line executions run via `backend/cli.py`.

### 1. List Available EEG/EOG/EMG Channels
Query the EDF file to check the label lists:
```sh
python backend/cli.py path/to/recording.edf --list-channels
```

### 2. Run YASA Autoscoring
```sh
python backend/cli.py path/to/recording.edf \
  --algorithm yasa \
  --eeg C3 \
  --out-dir outputs/
```

### 3. Run PhysioEx TinySleepNet with Multi-Channel Consensus
Run inference on multiple channels; probabilities will be averaged to create a consensus stage:
```sh
python backend/cli.py path/to/recording.edf \
  --algorithm tinysleepnet \
  --eeg C3,C4,F3,F4 \
  --out-dir outputs/
```

### 4. Run Luna POPS with Reference Channel
```sh
python backend/cli.py path/to/recording.edf \
  --algorithm luna \
  --eeg C3 \
  --ref M2 \
  --out-dir outputs/
```

### 5. Run Staging with SleepGPT Sequence Correction
```sh
python backend/cli.py path/to/recording.edf \
  --algorithm tinysleepnet \
  --sequence-correction sleepgpt \
  --eeg C3,C4 \
  --out-dir outputs/
```

### 6. Run the Legacy Tkinter GUI
To run the Tkinter desktop dashboard locally:
```sh
python backend/app.py
```

---

## 📥 Output JSON Formats

For a given `subject.edf`, the command-line utility outputs:

1.  `subject_<algorithm>.json`: Native JSON file compatible with ScoringNidra, containing lists of epochs and stage codes:
    *   `Wake` = 1
    *   `N1` = -1
    *   `N2` = -2
    *   `N3` = -3
    *   `REM` = 0
2.  `subject_<algorithm>_consensus_probabilities.json`: Matrix of raw probabilities (Wake, N1, N2, N3, REM) before SleepGPT correction.
3.  `subject_<algorithm>_per_channel_probabilities.json`: Individual per-channel model probabilities used to compute the consensus.
