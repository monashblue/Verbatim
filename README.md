# Verbatim
An Automatic Audio Transcriptor. Build by Claude AI, perfected with ChatGPT.

A desktop-based audio transcription application
with verbatim transcription, automatic speaker
diarization, timestamps, and unclear speech detection.
Powered by Faster-Whisper and Sherpa-ONNX.

**Audio Transcription & Speaker Diarization**

Project Verbatim is a desktop-based audio transcription application designed to produce detailed verbatim transcripts with automatic speaker identification.

Powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition and [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) for speaker diarization.

## Features

- **Verbatim Transcription** — Preserves filler words, repetitions, and spontaneous speech.
- **Speaker Diarization** — Automatically identifies and separates speakers (Speaker A, B, C, etc.).
- **Timestamps** — Adds timestamps to speaker turns and long speech segments.
- **Unclear Speech Detection** — Marks uncertain speech segments as `[tidak terdengar]`.
- **Terminology Correction** — Supports custom terminology and proper names.
- **Multilingual Support** — Supports multiple languages through Whisper.
- **Local Processing** — Audio is processed locally without requiring an online transcription service.

## Pros & Cons

### Pros
- Detailed verbatim transcription.
- Automatic speaker identification.
- Custom terminology support.
- Local audio processing.
- Simple graphical user interface.
- Multiple Whisper model sizes.

### Cons
- Transcription accuracy depends on audio quality.
- Speaker identification is not always perfect.
- Large AI models require additional storage.
- CPU processing can be slow for long recordings.
- Manual transcript verification is still necessary.

## Installation

### Option 1: Windows Executable (Recommended)

1. Download the latest `ProjectVerbatim.exe` from the [Releases](../../releases) page.
2. Extract the downloaded package.
3. Make sure the `models` directory is located alongside the executable.
4. Run `ProjectVerbatim.exe`.

No separate Python installation or manual Python dependency installation is required.

The Whisper model may need to be downloaded automatically on first use if it is not included in the package.

### Option 2: Run from Source

**Requirements:**
- Python 3.10 or newer (64-bit recommended).
- Internet connection for initial model downloads.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/project-verbatim.git
cd project-verbatim
```

Install dependencies:

```bash
python -m pip install faster-whisper==1.2.1 av==18.1.0 sherpa-onnx
```

Run the application:

```bash
python verbatim.py
```

## Models

Project Verbatim uses the following AI models:

| Model | Purpose |
|---|---|
| Faster-Whisper | Speech recognition |
| Pyannote Segmentation 3.0 | Speech segmentation |
| 3D-Speaker | Speaker embedding |

Required directory structure:

```text
Project Verbatim/
├── verbatim.py
├── assets/
│   └── logo.ico
└── models/
    ├── whisper/
    └── diarisasi/
        ├── segmentasi.onnx
        └── embedding.onnx
```

The Whisper model is downloaded automatically when it is not available locally.

Speaker diarization models must be provided separately in the `models/diarisasi` directory.

## Usage

1. Open Project Verbatim.
2. Select an audio file.
3. Choose the transcription language.
4. Select the Whisper model size.
5. Enable or disable speaker diarization.
6. Add custom terminology if necessary.
7. Click **Mulai Transkripsi**.
8. Wait until the transcription process is completed.

## Output

The application generates two text files:

**1. Transcript**

`filename_transkrip.txt`

Contains:
- Transcribed speech.
- Speaker labels.
- Timestamps.
- Unclear speech markers.

**2. Unclear Speech Notes**

`filename_tidak_terdengar.txt`

Contains:
- Timestamps of uncertain segments.
- Machine transcription guesses.
- Confidence scores.

## Disclaimer

AI-generated transcripts are not guaranteed to be perfectly accurate.

Always review the output against the original audio, especially for proper names, technical terminology, overlapping speech, and unclear recordings.

## License

See the LICENSE file for licensing information.
