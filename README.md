# Voicebox

All-in-one voice toolkit for Claude Code on Apple Silicon Macs.
Apple Silicon Mac 上的 Claude Code 一站式语音工具包。

## Features / 功能

- **Voice Design / 语音设计** — Create custom voices from text descriptions / 用文字描述创建自定义声音
- **Voice Cloning / 语音克隆** — Clone any voice from a 10s audio sample / 录 10 秒音频克隆任意声音
- **Text-to-Speech / 文字转语音** — Generate speech in 10 languages / 支持 10 种语言的语音合成
- **Multi-Speaker Conversations / 多角色对话** — Generate dialogues, dramas, news broadcasts / 生成对话、戏剧、新闻播报
- **Transcription / 语音转文字** — Speech-to-text for audio & video, 52 languages with auto-detection / 音频视频转文字，52 种语言自动识别

## Install / 安装

```bash
git clone https://github.com/tivojn/voicebox.git ~/.claude/skills/voicebox
```

That's it. Dependencies and models auto-download on first use.
就这样。依赖和模型首次使用时自动下载。

## Requirements / 环境要求

- Apple Silicon Mac (M1/M2/M3/M4)
- [Claude Code](https://claude.com/claude-code)
- [uv](https://docs.astral.sh/uv/) (usually pre-installed with Claude Code / 通常随 Claude Code 预装)
- ffmpeg — for recording & video transcription / 录音和视频转写需要 (`brew install ffmpeg`)

## Quick Start / 快速开始

```
/voicebox create a calm narrator voice profile          # Design a voice from description
/voicebox "Calm Narrator" "Hello, this is a test."      # Generate speech
/voicebox clone my voice                                # Record from mic & clone
/voicebox clone my voice from /path/to/audio.wav        # Clone from audio file
/voicebox transcribe /path/to/audio.wav                 # Transcribe audio/video
/voicebox create a news broadcast with anchor and reporter  # Multi-speaker conversation
```

## Voice Profiles / 语音档案

Profiles are stored in `data/profiles.json`. Two types:

- **Designed** — Created from a text description (supports `--instruct` style overrides)
- **Cloned** — Created from a real audio sample (reproduces the tone/energy of the original recording)

Start with no profiles — create your own with `/voicebox create ...` or `/voicebox clone ...`.

## Quality Options / 质量选项

All commands default to **high** (1.7B models). Use `--quality standard` for faster 0.6B models if needed.

| Category / 类别 | High (default) | Standard |
|-----------------|---------------|----------|
| Voice Design / 语音设计 | 1.7B | (same) |
| Voice Clone / 语音克隆 | 1.7B (~3.5GB) | 0.6B (~1.5GB) |
| Transcription / 语音转文字 | 1.7B (~3.5GB) | 0.6B (~1.5GB) |

## Recording / 录音

When recording from the mic, the script:
- Records for the specified duration (default 10 seconds)
- Prints `DONE — recording complete!` and plays a system bell when finished
- Auto-transcribes the recording using Qwen3-ASR
- Creates a cloned voice profile automatically

## Supported Languages / 支持语言

**TTS:** English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**ASR:** 52 languages with auto-detection / 52 种语言自动识别

## Script Commands / 命令参考

```bash
# List profiles
uv run ~/.claude/skills/voicebox/scripts/voicebox.py list

# Generate speech
uv run ~/.claude/skills/voicebox/scripts/voicebox.py generate "Profile" "text" --play

# Generate with standard quality (faster, less RAM)
uv run ~/.claude/skills/voicebox/scripts/voicebox.py generate "Profile" "text" --quality standard --play

# Create designed voice
uv run ~/.claude/skills/voicebox/scripts/voicebox.py create-designed "Name" --desc "description" --lang en

# Clone from audio file
uv run ~/.claude/skills/voicebox/scripts/voicebox.py create-cloned "Name" --audio /path/to.wav --ref-text "transcript" --lang en

# Record from mic and clone
uv run ~/.claude/skills/voicebox/scripts/voicebox.py record "Name" --duration 10 --lang en

# Transcribe audio/video
uv run ~/.claude/skills/voicebox/scripts/transcribe.py /path/to/file.wav

# Multi-speaker conversation from JSON script
uv run ~/.claude/skills/voicebox/scripts/voicebox.py conversation /tmp/script.json --play

# Delete a profile
uv run ~/.claude/skills/voicebox/scripts/voicebox.py delete "Name"
```
