# ccxa — Chichikusa Japanese Voice Chat Assistant

[日本語](./README.ja.md) | English

A locally-running Japanese voice chat assistant for Mac mini M2. All components (STT/LLM/TTS) run on-device with no cloud dependency, ensuring privacy while enabling real-time conversation.

## Tech Stack

| Component | Technology |
|-----------|------------|
| STT | [lilfugu-8bit](https://huggingface.co/holotherapper/lilfugu-8bit) (Qwen3-ASR via [mlx-audio](https://github.com/Blaizzy/mlx-audio)) |
| LLM | [Ternary Bonsai 8B 1.58bit](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit) ([mlx_lm.server](https://github.com/ml-explore/mlx-lm) + OpenAI API) |
| TTS | macOS `say` (system voice) |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) |
| Wake word | VAD + STT keyword match |
| Noise gate | [scipy](https://scipy.org/) Butterworth bandpass + amplitude gate |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Web search | [ddgs](https://github.com/deedy5/duckduckgo_search) (DuckDuckGo) |

## Features

- Wake word "チチクサ" starts conversation (chime response)
- Real-time conversation with VAD-based speech detection
- Short responses by default; say "詳しく" for detailed answers
- "〜を調べて" triggers DuckDuckGo web search
- Time/date queries answered from OS clock
- Weather via wttr.in API (Kusatsu, Kyoto, Tokyo, etc.)
- Exchange rates via Frankfurter API (USD, EUR, etc.)
- Barge-in support (interrupt bot speech)
- Auto-standby after 30 seconds of silence
- Say "さようなら" to end conversation

## Requirements

- macOS (Apple Silicon M2 or later recommended)
- Python 3.12+
- Microphone input device
- ~5 GB disk space (for model downloads)

## Setup

```fish
python3.12 -m venv .venv
source .venv/bin/activate.fish
pip install -e .
```

Models (lilfugu-8bit ~2.8 GB, Bonsai 8B ~2 GB) are downloaded from HuggingFace automatically on first run.

## Usage

```fish
python -m ccxa                 # Normal startup
python -m ccxa -v              # With debug logging
python -m ccxa --list-devices  # List audio devices
```

## Configuration

Edit `config.yaml` to adjust:

- Audio device, noise gate threshold, barge-in sensitivity
- VAD threshold and silence duration
- Wake word phrase
- STT / LLM models
- TTS voice and speed
- Conversation timeout

## License

MIT
