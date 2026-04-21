# ccxa — チチクサ 日本語音声チャットアシスタント

Mac mini M2 (16GB) 上でローカル実行する日本語音声チャットボット。STT/LLM/TTS すべてローカルモデルで動作し、プライバシーを保ちつつリアルタイムな会話を実現します。

## 技術スタック

| コンポーネント | 技術 |
|---|---|
| STT | [lilfugu-8bit](https://huggingface.co/holotherapper/lilfugu-8bit) (Qwen3-ASR, [mlx-audio](https://github.com/Blaizzy/mlx-audio)) |
| LLM | [Ternary Bonsai 8B 1.58bit](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit) ([mlx-lm](https://github.com/ml-explore/mlx-lm)) |
| TTS | macOS say (システムデフォルト音声) |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) |
| ウェイクワード | VAD + STT キーワードマッチ |
| ノイズゲート | [scipy](https://scipy.org/) Butterworth bandpass + 振幅ゲート |
| 音声入力 | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Web検索 | [ddgs](https://github.com/deedy5/duckduckgo_search) (DuckDuckGo) |

## 機能

- ウェイクワード「チチクサ」で会話開始（チャイム音で応答）
- VAD ベースの発話区間検出によるリアルタイム会話
- 短い応答がデフォルト。「詳しく」で詳細な説明
- 「〜を調べて」で Web 検索
- 時刻・日付の質問に OS 時刻で回答
- 天気の質問に wttr.in API で回答（草津、京都、東京など）
- 為替レートの質問に Frankfurter API で回答（ドル、ユーロなど）
- ボット発話中のバージイン（割り込み停止）対応
- 30秒無発話で自動的に待機状態に復帰
- 「さようなら」で会話終了

## 必要環境

- macOS (Apple Silicon M2 以上推奨)
- Python 3.12
- マイク入力デバイス
- 約 5GB のディスク空き容量（モデルダウンロード用）

## セットアップ

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # bash/zsh
# source .venv/bin/activate.fish  # fish
pip install -e .
```

初回起動時に STT モデル (lilfugu-8bit, ~2.8GB) と LLM モデル (Bonsai 8B, ~2GB) が HuggingFace から自動ダウンロードされます。

## 使い方

```bash
python -m ccxa              # 通常起動
python -m ccxa -v           # デバッグログ付き
python -m ccxa --list-devices  # 音声デバイス一覧
```

## 設定

`config.yaml` で各種パラメータを調整できます:

- 音声デバイス、ノイズゲート閾値、バージイン感度
- VAD 閾値、無音判定時間
- ウェイクワードフレーズ
- STT / LLM モデル
- TTS 音声、速度
- 会話タイムアウト時間

## ライセンス

MIT
