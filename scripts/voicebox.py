# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-audio>=0.3.1",
#     "soundfile",
#     "click",
# ]
#
# [tool.uv]
# prerelease = "allow"
# ///
"""
Voicebox — Standalone TTS with voice design and voice cloning via mlx-audio.

Usage:
    uv run voicebox.py list
    uv run voicebox.py create-designed "Calm Narrator" --desc "calm middle-aged male..." --lang en
    uv run voicebox.py create-cloned "My Voice" --audio /path/to/sample.wav --ref-text "transcript" --lang en
    uv run voicebox.py generate "Calm Narrator" "Hello world" --play
    uv run voicebox.py conversation script.json --play --trim-silence
    uv run voicebox.py delete "Calm Narrator"
"""

import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import click

DATA_DIR = Path(__file__).parent.parent / "data"
PROFILES_FILE = DATA_DIR / "profiles.json"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS = {
    "voice_design": {
        "standard": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    },
    "voice_clone": {
        "standard": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "high": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    },
    "asr": {
        "standard": "Qwen/Qwen3-ASR-0.6B",
        "high": "Qwen/Qwen3-ASR-1.7B",
    },
}

QUALITY_HELP = "Quality tier: 'standard' (0.6B, faster, less RAM) or 'high' (1.7B, better quality, ~8GB+ RAM)"


MODEL_SIZES = {
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": "~3.5GB",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16": "~3.5GB",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16": "~1.5GB",
    "Qwen/Qwen3-ASR-1.7B": "~3.5GB",
    "Qwen/Qwen3-ASR-0.6B": "~1.5GB",
}


def is_model_cached(model_id):
    """Check if a HuggingFace model is already downloaded."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HF cache uses -- as separator: models--org--repo
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_dir / folder_name / "snapshots"
    return model_dir.exists() and any(model_dir.iterdir())


def get_model(category, quality):
    """Get model ID for a category and quality tier."""
    tier = MODELS[category]
    if quality in tier:
        return tier[quality]
    # Fall back to best available (e.g. voice_design only has standard)
    return tier["standard"]


def load_model_with_progress(model_id):
    """Load a model, showing download progress info on first use."""
    from mlx_audio.tts.utils import load_model

    if not is_model_cached(model_id):
        size = MODEL_SIZES.get(model_id, "~3GB")
        click.echo(f"First-time setup: downloading {model_id} ({size})...")
        click.echo("This is a one-time download — future runs will be instant.")
    else:
        click.echo(f"Loading {model_id}...")

    return load_model(model_id)


LANG_MAP = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


def load_profiles():
    if not PROFILES_FILE.exists():
        return {"profiles": []}
    return json.loads(PROFILES_FILE.read_text())


def save_profiles(data):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(data, indent=2) + "\n")


def find_profile(name):
    data = load_profiles()
    name_lower = name.lower()
    for p in data["profiles"]:
        if p["name"].lower() == name_lower:
            return p
    for p in data["profiles"]:
        if name_lower in p["name"].lower():
            return p
    return None


def slugify(name):
    # Keep Unicode letters and digits, replace separators with hyphens
    slug = re.sub(r"[^\w]+", "-", name.lower(), flags=re.UNICODE).strip("-")
    if not slug:
        # Fallback for names that produce empty slugs
        import hashlib
        slug = hashlib.md5(name.encode()).hexdigest()[:8]
    return slug


def collect_audio(results):
    """Collect audio from generation results, concatenating segments."""
    import numpy as np

    audio_parts = []
    sample_rate = None
    for r in results:
        if r.audio is not None:
            audio_parts.append(np.array(r.audio))
            if sample_rate is None:
                sample_rate = r.sample_rate

    if not audio_parts:
        return None, None

    audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    return audio, sample_rate


@click.group()
def cli():
    """Voicebox — TTS with voice design and voice cloning."""
    pass


@cli.command("list")
def list_profiles():
    """List all voice profiles."""
    data = load_profiles()
    if not data["profiles"]:
        click.echo("No profiles. Create one with 'create-designed' or 'create-cloned'.")
        return
    for p in data["profiles"]:
        kind = p["type"]
        if kind == "designed":
            click.echo(f"  {p['name']}  (designed, {p['language']})  — {p['description'][:60]}...")
        else:
            click.echo(f"  {p['name']}  (cloned, {p['language']})  — ref: {p['ref_audio']}")


@cli.command("models")
def list_models():
    """List available models and quality tiers."""
    click.echo("Available models:\n")
    click.echo("Voice Design (create-designed):")
    click.echo(f"  standard (default): {MODELS['voice_design']['standard']}")
    click.echo(f"  (Only 1.7B available — always best quality)\n")
    click.echo("Voice Clone (generate with cloned profiles, create-cloned, record):")
    for q, m in MODELS["voice_clone"].items():
        default = " (default)" if q == "standard" else ""
        click.echo(f"  {q}{default}: {m}")
    click.echo()
    click.echo("Speech Recognition / Transcription (record auto-transcribe, transcribe):")
    for q, m in MODELS["asr"].items():
        default = " (default)" if q == "standard" else ""
        click.echo(f"  {q}{default}: {m}")
    click.echo()
    click.echo("Use --quality high on any command to upgrade to 1.7B models.")
    click.echo("Note: 'high' models need ~8GB+ RAM and download ~3GB on first use.")


@cli.command("create-designed")
@click.argument("name")
@click.option("--desc", required=True, help="Voice description for the VoiceDesign model")
@click.option("--lang", default="en", help="Language code (en, zh, ja, ko, de, fr, ru, pt, es, it)")
@click.option("--sample-text", default=None, help="Text to synthesize as reference sample")
@click.option("--quality", default="high", type=click.Choice(["standard", "high"]), help=QUALITY_HELP)
def create_designed(name, desc, lang, sample_text, quality):
    """Create a voice profile from a text description (VoiceDesign model)."""
    if find_profile(name):
        click.echo(f"Profile '{name}' already exists.", err=True)
        sys.exit(1)

    if sample_text is None:
        sample_text = (
            "The morning sun rose gently over the quiet village, casting golden light "
            "across the cobblestone streets. Birds sang their morning songs as the world "
            "slowly came to life."
        )

    model_id = get_model("voice_design", quality)
    click.echo(f"Creating designed voice: {name}")
    click.echo(f"Description: {desc}")

    import soundfile as sf

    model = load_model_with_progress(model_id)
    language = LANG_MAP.get(lang, "English")

    slug = slugify(name)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    click.echo("Generating reference sample...")
    results = list(model.generate_voice_design(
        text=sample_text,
        language=language,
        instruct=desc,
    ))

    audio, sample_rate = collect_audio(results)
    if audio is None:
        click.echo("Error: No audio generated", err=True)
        sys.exit(1)

    sample_path = SAMPLES_DIR / f"{slug}.wav"
    sf.write(str(sample_path), audio, sample_rate)

    info = sf.info(str(sample_path))
    click.echo(f"Saved reference sample: {sample_path} ({info.duration:.1f}s)")

    # Register profile
    data = load_profiles()
    data["profiles"].append({
        "id": slug,
        "name": name,
        "type": "designed",
        "description": desc,
        "language": lang,
        "sample_audio": f"samples/{slug}.wav",
        "sample_text": sample_text,
    })
    save_profiles(data)
    click.echo(f"Profile '{name}' created.")


@cli.command("create-cloned")
@click.argument("name")
@click.option("--audio", required=True, type=click.Path(exists=True), help="Path to reference audio WAV file")
@click.option("--ref-text", required=True, help="Transcript of the reference audio")
@click.option("--lang", default="en", help="Language code")
def create_cloned(name, audio, ref_text, lang):
    """Create a voice profile from an audio sample (Base model for cloning)."""
    if find_profile(name):
        click.echo(f"Profile '{name}' already exists.", err=True)
        sys.exit(1)

    # Copy audio to samples dir
    slug = slugify(name)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAMPLES_DIR / f"{slug}.wav"
    src = Path(audio).resolve()
    if src != dest.resolve():
        shutil.copy2(audio, dest)
        click.echo(f"Copied reference audio to {dest}")
    else:
        click.echo(f"Reference audio already at {dest}")

    # Register profile
    data = load_profiles()
    data["profiles"].append({
        "id": slug,
        "name": name,
        "type": "cloned",
        "ref_audio": f"samples/{slug}.wav",
        "ref_text": ref_text,
        "language": lang,
    })
    save_profiles(data)
    click.echo(f"Profile '{name}' created (cloned).")


@cli.command("generate")
@click.argument("profile_name")
@click.argument("text")
@click.option("--instruct", default=None, help="Style/emotion instruction (designed voices)")
@click.option("--output", "-o", default="/tmp/voicebox_output", help="Output file path (without extension)")
@click.option("--play/--no-play", default=False, help="Play audio after generation")
@click.option("--quality", default="high", type=click.Choice(["standard", "high"]), help=QUALITY_HELP)
def generate(profile_name, text, instruct, output, play, quality):
    """Generate speech using a voice profile."""
    profile = find_profile(profile_name)
    if not profile:
        click.echo(f"Profile '{profile_name}' not found. Run 'list' to see available profiles.", err=True)
        sys.exit(1)

    click.echo(f"Using profile: {profile['name']} ({profile['type']})")

    import soundfile as sf

    if profile["type"] == "designed":
        voice_desc = instruct if instruct else profile["description"]
        model_id = get_model("voice_design", quality)
        model = load_model_with_progress(model_id)
        language = LANG_MAP.get(profile["language"], "English")
        click.echo("Generating audio...")
        results = list(model.generate_voice_design(
            text=text,
            language=language,
            instruct=voice_desc,
        ))

    elif profile["type"] == "cloned":
        ref_audio_path = str(DATA_DIR / profile["ref_audio"])
        model_id = get_model("voice_clone", quality)
        model = load_model_with_progress(model_id)
        click.echo("Generating audio (voice cloning)...")
        results = list(model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=profile["ref_text"],
        ))
    else:
        click.echo(f"Unknown profile type: {profile['type']}", err=True)
        sys.exit(1)

    audio, sample_rate = collect_audio(results)
    if audio is None:
        click.echo("Error: No audio generated", err=True)
        sys.exit(1)

    out_path = f"{output}.wav"
    sf.write(out_path, audio, sample_rate)

    info = sf.info(out_path)
    click.echo(f"Saved: {out_path} ({info.duration:.1f}s)")

    if play:
        click.echo("Playing...")
        subprocess.run(["afplay", out_path])


@cli.command("record")
@click.argument("name")
@click.option("--duration", "-d", default=10, help="Recording duration in seconds")
@click.option("--lang", default="en", help="Language code")
@click.option("--ref-text", default=None, help="Transcript of what you'll say (if known)")
@click.option("--quality", default="high", type=click.Choice(["standard", "high"]), help=QUALITY_HELP)
def record_and_clone(name, duration, lang, ref_text, quality):
    """Record from microphone and create a cloned voice profile."""
    if find_profile(name):
        click.echo(f"Profile '{name}' already exists.", err=True)
        sys.exit(1)

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        click.echo("Error: ffmpeg not found. Install with: brew install ffmpeg", err=True)
        sys.exit(1)

    slug = slugify(name)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rec_path = SAMPLES_DIR / f"{slug}.wav"

    # Record from microphone
    click.echo(f"Recording {duration}s from microphone — speak now!")
    click.echo("=" * 40)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-i", ":default",
            "-t", str(duration),
            "-ar", "24000",
            "-ac", "1",
            str(rec_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo(f"Recording failed: {result.stderr}", err=True)
        sys.exit(1)

    # Clear notification that recording has stopped
    click.echo("=" * 40)
    click.echo("DONE — recording complete! You can stop speaking.")
    # System bell to audibly notify the user
    click.echo("\a", nl=False)

    import soundfile as sf
    info = sf.info(str(rec_path))
    click.echo(f"Recorded: {rec_path} ({info.duration:.1f}s)")

    if not ref_text:
        # Auto-transcribe using built-in transcribe.py
        asr_model = get_model("asr", quality)
        if not is_model_cached(asr_model):
            size = MODEL_SIZES.get(asr_model, "~3GB")
            click.echo(f"First-time setup: downloading ASR model {asr_model} ({size})...")
            click.echo("This is a one-time download — future runs will be instant.")
        click.echo(f"Auto-transcribing recording (ASR: {asr_model})...")
        transcribe_script = Path(__file__).parent / "transcribe.py"
        lang_flag = ["--language", lang] if lang else []
        tr_result = subprocess.run(
            ["uv", "run", str(transcribe_script), str(rec_path),
             "--model", asr_model] + lang_flag,
            stdout=subprocess.PIPE, stderr=None, text=True,
        )
        if tr_result.returncode == 0 and tr_result.stdout.strip():
            ref_text = tr_result.stdout.strip()
            click.echo(f"Transcript: {ref_text}")
        else:
            click.echo("Auto-transcription failed. Provide transcript manually:", err=True)
            click.echo(f'  uv run voicebox.py create-cloned "{name}" --audio {rec_path} --ref-text "what you said" --lang {lang}')
            return

    # Create cloned profile
    data = load_profiles()
    data["profiles"].append({
        "id": slug,
        "name": name,
        "type": "cloned",
        "ref_audio": f"samples/{slug}.wav",
        "ref_text": ref_text,
        "language": lang,
    })
    save_profiles(data)
    click.echo(f"Profile '{name}' created (cloned).")


@cli.command("delete")
@click.argument("name")
def delete_profile(name):
    """Delete a voice profile."""
    data = load_profiles()
    target = None
    for p in data["profiles"]:
        if p["name"].lower() == name.lower():
            target = p
            break
    if not target:
        click.echo(f"Profile '{name}' not found.", err=True)
        sys.exit(1)

    # Remove sample file if exists
    for key in ("sample_audio", "ref_audio"):
        if key in target:
            sample_path = DATA_DIR / target[key]
            if sample_path.exists():
                sample_path.unlink()
                click.echo(f"Deleted {sample_path}")

    data["profiles"] = [p for p in data["profiles"] if p is not target]
    save_profiles(data)
    click.echo(f"Profile '{target['name']}' deleted.")


def trim_silence(input_path, output_path):
    """Trim leading and trailing silence from a WAV file using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af",
            "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB,"
            "areverse",
            "-ar", "24000", "-ac", "1",
            str(output_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo(f"  Warning: ffmpeg trim failed, using untrimmed: {result.stderr[:200]}", err=True)
        shutil.copy2(input_path, output_path)


def normalize_wav(input_path, output_path):
    """Normalize a WAV to 24kHz mono 16-bit using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
            str(output_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo(f"  Warning: ffmpeg normalize failed: {result.stderr[:200]}", err=True)
        shutil.copy2(input_path, output_path)


def combine_wavs(wav_paths, output_path, gap_seconds=0.25):
    """Combine multiple WAV files with silence gaps using the wave module.

    All input WAVs must be 24kHz mono 16-bit PCM.
    """
    sample_rate = 24000
    sample_width = 2  # 16-bit
    n_channels = 1
    silence_frames = int(sample_rate * gap_seconds)
    silence_data = b"\x00" * (silence_frames * sample_width * n_channels)

    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)

        for i, wav_path in enumerate(wav_paths):
            with wave.open(str(wav_path), "rb") as inp:
                out.writeframes(inp.readframes(inp.getnframes()))
            if i < len(wav_paths) - 1:
                out.writeframes(silence_data)


def get_wav_duration(wav_path):
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_path), "rb") as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return 0.0


@cli.command("conversation")
@click.argument("script_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Output directory for segments and combined WAV")
@click.option("--gap", default=None, type=float, help="Silence gap between segments in seconds (overrides script)")
@click.option("--quality", default="high", type=click.Choice(["standard", "high"]), help=QUALITY_HELP)
@click.option("--trim-silence/--no-trim-silence", "do_trim", default=True, help="Trim leading/trailing silence from segments (default: on)")
@click.option("--play/--no-play", default=False, help="Play combined result after generation")
def conversation(script_file, output_dir, gap, quality, do_trim, play):
    """Generate a multi-speaker conversation from a JSON script.

    The JSON script should contain:

    \b
    {
      "title": "My Show",
      "gap": 0.25,
      "lines": [
        {"profile": "News Anchor", "text": "Good evening.", "instruct": "serious"},
        {"profile": "Reporter", "text": "Thanks, Tom!"}
      ]
    }

    Each line uses an existing voice profile. The "instruct" field is optional
    and only applies to designed (not cloned) profiles.
    """
    # Check for ffmpeg if trimming
    if do_trim and not shutil.which("ffmpeg"):
        click.echo("Error: ffmpeg not found (needed for --trim-silence). Install: brew install ffmpeg", err=True)
        click.echo("Or use --no-trim-silence to skip trimming.", err=True)
        sys.exit(1)

    # Load and validate script
    script_path = Path(script_file)
    script = json.loads(script_path.read_text())

    title = script.get("title", "conversation")
    lines = script.get("lines", [])
    if not lines:
        click.echo("Error: Script has no lines.", err=True)
        sys.exit(1)

    # Gap: CLI flag overrides script value, default 0.25
    gap_seconds = gap if gap is not None else script.get("gap", 0.25)

    # Validate all profiles upfront
    click.echo(f"Script: {title} ({len(lines)} lines, gap={gap_seconds}s)")
    profiles_used = {}
    for i, line in enumerate(lines):
        pname = line.get("profile", "")
        if pname not in profiles_used:
            profile = find_profile(pname)
            if not profile:
                click.echo(f"Error: Profile '{pname}' not found (line {i+1}). Run 'list' to see available profiles.", err=True)
                sys.exit(1)
            profiles_used[pname] = profile
            click.echo(f"  Profile: {profile['name']} ({profile['type']})")

    # Set up output directory
    if output_dir is None:
        title_slug = slugify(title)
        output_dir = f"/tmp/voicebox_{title_slug}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {out_path}")

    # Load models — cache by model_id to avoid reloading
    import soundfile as sf

    loaded_models = {}

    def get_or_load_model(profile, qual):
        if profile["type"] == "designed":
            mid = get_model("voice_design", qual)
        else:
            mid = get_model("voice_clone", qual)
        if mid not in loaded_models:
            loaded_models[mid] = load_model_with_progress(mid)
        return loaded_models[mid], mid

    # Generate each line
    segment_paths = []
    click.echo(f"\nGenerating {len(lines)} segments...")

    for i, line in enumerate(lines):
        pname = line["profile"]
        text = line["text"]
        instruct = line.get("instruct", None)
        profile = profiles_used[pname]
        num = f"{i+1:03d}"
        seg_name = f"{num}_{slugify(profile['name'])}"

        click.echo(f"\n[{i+1}/{len(lines)}] {profile['name']}: {text[:60]}{'...' if len(text) > 60 else ''}")

        model, model_id = get_or_load_model(profile, quality)

        if profile["type"] == "designed":
            voice_desc = instruct if instruct else profile["description"]
            language = LANG_MAP.get(profile["language"], "English")
            results = list(model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_desc,
            ))
        elif profile["type"] == "cloned":
            ref_audio_path = str(DATA_DIR / profile["ref_audio"])
            results = list(model.generate(
                text=text,
                ref_audio=ref_audio_path,
                ref_text=profile["ref_text"],
            ))
        else:
            click.echo(f"  Error: Unknown profile type '{profile['type']}', skipping.", err=True)
            continue

        audio, sample_rate = collect_audio(results)
        if audio is None:
            click.echo(f"  Error: No audio generated for line {i+1}, skipping.", err=True)
            continue

        raw_path = out_path / f"{seg_name}_raw.wav"
        sf.write(str(raw_path), audio, sample_rate)
        raw_dur = get_wav_duration(raw_path)

        if do_trim:
            trimmed_path = out_path / f"{seg_name}.wav"
            trim_silence(raw_path, trimmed_path)
            trimmed_dur = get_wav_duration(trimmed_path)
            click.echo(f"  Saved: {trimmed_path.name} ({trimmed_dur:.1f}s, trimmed from {raw_dur:.1f}s)")
            raw_path.unlink()  # Clean up raw file
            segment_paths.append(trimmed_path)
        else:
            # Still normalize to consistent format for combining
            final_path = out_path / f"{seg_name}.wav"
            normalize_wav(raw_path, final_path)
            final_dur = get_wav_duration(final_path)
            click.echo(f"  Saved: {final_path.name} ({final_dur:.1f}s)")
            raw_path.unlink()
            segment_paths.append(final_path)

    if not segment_paths:
        click.echo("\nError: No segments were generated.", err=True)
        sys.exit(1)

    # Combine all segments
    combined_path = out_path / f"{slugify(title)}_combined.wav"
    click.echo(f"\nCombining {len(segment_paths)} segments (gap={gap_seconds}s)...")
    combine_wavs(segment_paths, combined_path, gap_seconds)
    combined_dur = get_wav_duration(combined_path)

    # Summary
    click.echo(f"\n{'=' * 50}")
    click.echo(f"Conversation: {title}")
    click.echo(f"{'=' * 50}")
    total_seg_dur = 0.0
    for sp in segment_paths:
        dur = get_wav_duration(sp)
        total_seg_dur += dur
        click.echo(f"  {sp.name:40s} {dur:5.1f}s")
    click.echo(f"{'─' * 50}")
    click.echo(f"  Segments total: {total_seg_dur:.1f}s")
    click.echo(f"  Gaps: {len(segment_paths)-1} × {gap_seconds}s = {(len(segment_paths)-1)*gap_seconds:.1f}s")
    click.echo(f"  Combined: {combined_path.name:30s} {combined_dur:.1f}s")
    click.echo(f"  Output dir: {out_path}")

    if play:
        click.echo("\nPlaying combined audio...")
        subprocess.run(["afplay", str(combined_path)])


if __name__ == "__main__":
    cli()
