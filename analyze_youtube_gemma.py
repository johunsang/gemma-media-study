import argparse
import base64
import html
import io
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Analyze this media. Write your entire response in the SAME language as the "
    "spoken content / transcript (English for an English video, Korean for a Korean "
    "video, and so on). Summarize the main topic, key points, important visual "
    "details, any spoken content from the transcript, and provide a short list of "
    "actionable study insights."
)

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:12b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video and study it with a local Ollama Gemma 4 model."
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "--local-video",
        help="Analyze an existing local video file instead of downloading from YouTube.",
    )
    parser.add_argument(
        "--local-media",
        help="Analyze an existing local video or audio file instead of downloading from YouTube.",
    )
    parser.add_argument(
        "--ollama-endpoint",
        default=DEFAULT_ENDPOINT,
        help="Base URL of the local Ollama server. Default: http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        "--ollama-model",
        dest="model",
        default=DEFAULT_MODEL,
        help="Ollama model tag used for analysis, frame description, and audio ASR.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory where the downloaded video and analysis are saved.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=480,
        help="Maximum video height to download.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens (num_predict) to generate for the analysis.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Analysis prompt passed to Gemma.",
    )
    parser.add_argument(
        "--output-language",
        default="auto",
        help=(
            "Language for the analysis and frame summaries. 'auto' matches the "
            "content's language; otherwise force a language (e.g. 'Korean') even for "
            "English content. Verbatim subtitles always stay in the original language."
        ),
    )
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Enable Gemma thinking mode (Ollama 'think' option).",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=32,
        help=(
            "Number of frames to extract from a video for the frame-by-frame "
            "breakdown Markdown. Set 0 to disable. Audio inputs are skipped."
        ),
    )
    parser.add_argument(
        "--analysis-frames",
        type=int,
        default=6,
        help="How many sampled frames to attach as images to the overall analysis call.",
    )
    parser.add_argument(
        "--json-summary-path",
        help="Optional path where a machine-readable result summary is written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def ollama_chat(
    endpoint: str,
    model: str,
    text: str,
    images: list[str] | None = None,
    num_predict: int = 1024,
    keep_thinking: bool = False,
) -> str:
    """Call the local Ollama /api/chat endpoint and return the assistant text.

    ``images`` is a list of base64-encoded image bytes (no data: prefix), which
    Ollama feeds to the model's vision tower.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed. Run: pip install -r requirements.txt"
        ) from exc

    url = endpoint.rstrip("/") + "/api/chat"
    message: dict[str, Any] = {"role": "user", "content": text}
    if images:
        message["images"] = images

    payload: dict[str, Any] = {
        "model": model,
        "messages": [message],
        "stream": False,
        "think": bool(keep_thinking),
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_predict": num_predict,
        },
    }

    response = requests.post(url, json=payload, timeout=900)
    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama returned 404 for model '{model}'. Pull it first: ollama pull {model}"
        )
    response.raise_for_status()
    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# YouTube / media download
# ---------------------------------------------------------------------------

def download_youtube_video(url: str, output_dir: Path, max_height: int) -> Path:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import YoutubeDLError
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    # Give every video its own folder so the mp4, subtitles, thumbnail, analysis
    # Markdown, and extracted frames all stay grouped together.
    folder = "%(title).80s-%(id)s"
    output_template = str(output_dir / folder / f"{folder}.%(ext)s")

    # Prefer already-merged (progressive) streams so no ffmpeg merge step is needed.
    ydl_opts: dict[str, Any] = {
        "format": (
            f"best[ext=mp4][vcodec!=none][acodec!=none][height<={max_height}]/"
            f"best[vcodec!=none][acodec!=none][height<={max_height}]/"
            "best[ext=mp4][vcodec!=none][acodec!=none]/"
            "best[vcodec!=none][acodec!=none]"
        ),
        "noplaylist": True,
        "outtmpl": output_template,
        "restrictfilenames": True,
        "subtitlesformat": "vtt",
        # Fetch the original-language captions (any language) plus English, so a
        # Korean video keeps Korean subtitles, a Japanese video keeps Japanese, etc.
        "subtitleslangs": [
            "en.*", "ko.*", "ja.*", "zh.*", "es.*", "fr.*", "de.*", "vi.*",
            "id.*", "pt.*", "ru.*", "ar.*", "hi.*", "th.*", "it.*", "tr.*",
            "en", "ko", "ja", "zh", "es", "fr", "de", "vi",
        ],
        "writeautomaticsub": True,
        "writesubtitles": True,
        "writethumbnail": True,
        "windowsfilenames": True,
        # Be gentle so YouTube is less likely to rate-limit (HTTP 429).
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "sleep_interval_requests": 1,
        # A missing/failed subtitle or thumbnail must NOT abort the video download.
        "ignoreerrors": False,
    }

    def run(opts: dict[str, Any]) -> Path:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("yt-dlp did not return video metadata.")
            requested_downloads = info.get("requested_downloads") or []
            if requested_downloads:
                filepath = requested_downloads[0].get("filepath")
                if filepath:
                    return Path(filepath)
            filename = ydl.prepare_filename(info)
            merged = Path(filename).with_suffix(".mp4")
            return merged if merged.exists() else Path(filename)

    try:
        return run(ydl_opts)
    except YoutubeDLError as exc:
        message = str(exc).lower()
        # Subtitles are optional. If they get rate-limited (429) or otherwise fail,
        # retry the video download without them rather than losing the whole video.
        if "subtitle" in message or "429" in message or "too many requests" in message:
            print(f"Subtitle/thumbnail download failed ({exc}); retrying without subtitles...")
            fallback = {
                **ydl_opts,
                "writesubtitles": False,
                "writeautomaticsub": False,
            }
            return run(fallback)
        raise


def media_kind_from_path(path: Path) -> str:
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
    if path.suffix.lower() in audio_extensions:
        return "audio"
    return "video"


# ---------------------------------------------------------------------------
# Frame extraction (PyAV, no ffmpeg CLI)
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: Path,
    count: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Extract ``count`` frames evenly across the whole video using PyAV."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required for frame extraction. Run: pip install -r requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
    except (IndexError, KeyError):
        container.close()
        return []
    stream.thread_type = "AUTO"

    time_base = stream.time_base
    duration = None
    if stream.duration and time_base:
        duration = float(stream.duration * time_base)
    elif container.duration:
        duration = float(container.duration) / av.time_base

    # Midpoint of each of ``count`` equal slices, avoiding the first/last frames.
    if duration and duration > 0:
        targets = [duration * (i + 0.5) / count for i in range(count)]
    else:
        targets = [None] * count

    frames: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        try:
            if target is not None and time_base:
                container.seek(
                    int(target / time_base), stream=stream, any_frame=False, backward=True
                )
            decoded = next(container.decode(video=0), None)
            if decoded is None:
                continue
            image = decoded.to_image()
            if image.mode != "RGB":
                image = image.convert("RGB")
            frame_path = output_dir / f"frame_{index:04d}.jpg"
            image.save(frame_path, "JPEG", quality=90)
            frame_time = (
                float(decoded.pts * time_base)
                if decoded.pts and time_base
                else (target or 0.0)
            )
            frames.append({"path": frame_path, "time": frame_time})
        except Exception as exc:
            print(f"Frame {index} extraction failed: {exc}")
            continue

    container.close()
    return frames


# ---------------------------------------------------------------------------
# Audio (Gemma 4 native ASR via Ollama — audio bytes go in the "images" field)
# ---------------------------------------------------------------------------

def ollama_unload(endpoint: str, model: str) -> None:
    """Ask Ollama to unload a model from VRAM (keep_alive=0) so Whisper has room."""
    try:
        import requests
        requests.post(
            endpoint.rstrip("/") + "/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


def transcribe_with_whisper(
    media_path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Transcribe media with faster-whisper — accurate verbatim subtitles with
    proper segment timestamps, language auto-detected. Returns (cues, language).

    The model is loaded and released per call so it does not sit in VRAM next to
    Gemma (the two together overflow a small GPU)."""
    import gc

    from faster_whisper import WhisperModel

    try:
        whisper = WhisperModel(model_size, device="cuda", compute_type="float16")
        print(f"Whisper '{model_size}' loaded on GPU.")
    except Exception as exc:
        print(f"Whisper GPU unavailable ({exc}); using CPU.")
        whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
        segments, info = whisper.transcribe(
            str(media_path), language=language, vad_filter=True, beam_size=5
        )
        cues: list[dict[str, Any]] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            cues.append(_make_cue(len(cues), float(seg.start), float(seg.end), text))
        detected = getattr(info, "language", None)
    finally:
        del whisper
        gc.collect()
    return cues, detected


def media_duration_seconds(media_path: Path) -> float | None:
    try:
        import av
    except ImportError:
        return None
    try:
        container = av.open(str(media_path))
        duration = float(container.duration) / av.time_base if container.duration else None
        container.close()
        return duration
    except Exception:
        return None


def extract_audio_wav(media_path: Path, start: float, duration: float) -> bytes:
    """Extract a mono 16kHz WAV clip [start, start+duration] as bytes (PyAV)."""
    import av

    container = av.open(str(media_path))
    try:
        astream = container.streams.audio[0]
    except (IndexError, KeyError):
        container.close()
        return b""

    buffer = io.BytesIO()
    out = av.open(buffer, mode="w", format="wav")
    ostream = out.add_stream("pcm_s16le", rate=16000)
    ostream.layout = "mono"
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    if start > 0 and astream.time_base:
        try:
            container.seek(int(start / astream.time_base), stream=astream, backward=True)
        except Exception:
            pass

    end = start + duration
    for frame in container.decode(astream):
        t = float(frame.time) if frame.time is not None else 0.0
        if t < start:
            continue
        if t > end:
            break
        frame.pts = None
        for rframe in resampler.resample(frame):
            for packet in ostream.encode(rframe):
                out.mux(packet)
    for packet in ostream.encode(None):
        out.mux(packet)
    out.close()
    container.close()
    return buffer.getvalue()


def transcribe_audio_segments(
    endpoint: str,
    model: str,
    media_path: Path,
    keep_thinking: bool,
    chunk_seconds: float = 30.0,
    max_chunks: int = 40,
) -> list[dict[str, Any]]:
    """Summarize the spoken content of the media in <=30s chunks using Gemma 4's
    native audio understanding. Returns subtitle-like cues, one per chunk."""
    duration = media_duration_seconds(media_path) or chunk_seconds
    count = min(max_chunks, max(1, int(math.ceil(duration / chunk_seconds))))
    print(f"Transcribing audio in {count} chunk(s) of {int(chunk_seconds)}s...")

    # Verbatim ASR — exact transcription, NOT a summary, so the subtitles match the
    # speech word-for-word (per the Gemma 4 audio ASR best-practice prompt).
    prompt = (
        "Transcribe this speech segment exactly, word for word, in the SAME language "
        "as the speech. Output ONLY the transcription text — no summary, no notes, no "
        "translation. When transcribing numbers, write digits (e.g. 1.7, not 'one "
        "point seven'). If there is no speech, reply 'No speech.'"
    )
    cues: list[dict[str, Any]] = []
    for index in range(count):
        start = index * chunk_seconds
        try:
            wav = extract_audio_wav(media_path, start, chunk_seconds)
        except Exception as exc:
            print(f"Audio chunk {index} extraction failed: {exc}")
            continue
        if not wav:
            continue
        try:
            text = ollama_chat(
                endpoint, model, prompt,
                images=[base64.b64encode(wav).decode("ascii")],
                num_predict=300, keep_thinking=keep_thinking,
            )
        except Exception as exc:
            print(f"Audio chunk {index} analysis failed: {exc}")
            continue
        text = text.strip()
        if not text or text.lower().startswith("no speech"):
            continue
        cues.append(_make_cue(len(cues), start, start + chunk_seconds, text))
    return cues


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

def find_subtitle_file(video_path: Path) -> Path | None:
    candidates: list[Path] = []
    for suffix in ("vtt", "srt"):
        candidates.extend(video_path.parent.glob(f"{video_path.stem}*.{suffix}"))
    if not candidates:
        return None
    # Prefer the original-language captions. YouTube marks the source-language
    # auto-caption with a "-orig" suffix; that keeps a Korean video Korean, a
    # Japanese video Japanese, etc., instead of an English machine translation.
    originals = [p for p in candidates if "orig" in p.name.lower()]
    if originals:
        return sorted(originals)[0]
    return sorted(candidates)[0]


def find_thumbnail_file(video_path: Path) -> Path | None:
    candidates: list[Path] = []
    for suffix in ("jpg", "jpeg", "png", "webp"):
        candidates.extend(video_path.parent.glob(f"{video_path.stem}*.{suffix}"))
    return sorted(candidates)[0] if candidates else None


def parse_timestamp(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])


def clean_subtitle_text(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()

    words = text.split()
    if len(words) % 2 == 0:
        midpoint = len(words) // 2
        if words[:midpoint] == words[midpoint:]:
            return " ".join(words[:midpoint])
    return text


def parse_vtt(path: Path) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    block: list[str] = []

    def flush(current: list[str]) -> None:
        if not current:
            return
        timing_index = next((i for i, line in enumerate(current) if "-->" in line), -1)
        if timing_index == -1:
            return
        timing = current[timing_index]
        start_raw, end_raw = timing.split("-->", 1)
        end_raw = end_raw.split()[0]
        text = clean_subtitle_text(current[timing_index + 1 :])
        if not text:
            return
        cues.append(_make_cue(len(cues), parse_timestamp(start_raw), parse_timestamp(end_raw), text))

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == "WEBVTT" or stripped.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if not stripped:
            flush(block)
            block = []
            continue
        block.append(stripped)
    flush(block)
    return cues


def parse_srt(path: Path) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index == -1:
            continue
        start_raw, end_raw = lines[timing_index].split("-->", 1)
        text = clean_subtitle_text(lines[timing_index + 1 :])
        if not text:
            continue
        cues.append(_make_cue(len(cues), parse_timestamp(start_raw), parse_timestamp(end_raw), text))
    return cues


def _make_cue(index: int, start: float, end: float, text: str) -> dict[str, Any]:
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
    }


def parse_subtitle(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".srt":
        return parse_srt(path)
    return parse_vtt(path)


def merge_subtitle_cues(
    cues: list[dict[str, Any]],
    target_duration: float = 7.0,
    max_duration: float = 12.0,
    max_chars: int = 140,
    max_gap: float = 2.5,
) -> list[dict[str, Any]]:
    """Merge YouTube's tiny auto-caption fragments into scene-sized segments.

    Auto-captions arrive as 1-3 word cues (hundreds per video), which are useless
    for loop practice. We glue consecutive cues together until a segment reaches a
    scene-like length, preferring to break at sentence punctuation.
    """
    if not cues:
        return cues

    merged: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            cur["index"] = len(merged)
            merged.append(cur)
            cur = None

    for cue in cues:
        if cur is None:
            cur = _make_cue(len(merged), cue["start"], cue["end"], cue["text"])
            continue

        gap = cue["start"] - cur["end"]
        duration = cur["end"] - cur["start"]
        combined_len = len(cur["text"]) + 1 + len(cue["text"])
        ends_sentence = cur["text"].rstrip().endswith((".", "?", "!", "…"))

        long_enough = duration >= target_duration and ends_sentence
        too_long = duration >= max_duration or combined_len > max_chars
        big_gap = gap > max_gap

        if long_enough or too_long or big_gap:
            prev_text = cur["text"]
            flush()
            # Trim words the new scene repeats from the previous scene's tail.
            start_text = _strip_overlap(prev_text, cue["text"])
            cur = _make_cue(len(merged), cue["start"], cue["end"], start_text)
        else:
            # De-duplicate rolling-caption overlap before joining.
            cur["text"] = _join_caption(cur["text"], cue["text"])
            cur["end"] = cue["end"]

    flush()
    return merged


def _strip_overlap(prev_text: str, new_text: str) -> str:
    """Drop leading words of new_text that repeat the tail of prev_text."""
    prev_text = prev_text.strip()
    new_text = new_text.strip()
    if not prev_text or not new_text:
        return new_text
    a_words = prev_text.split()
    b_words = new_text.split()
    max_overlap = min(len(a_words), len(b_words))
    for k in range(max_overlap, 0, -1):
        if a_words[-k:] == b_words[:k]:
            return " ".join(b_words[k:]) or new_text
    return new_text


def _join_caption(existing: str, addition: str) -> str:
    """Join two caption fragments, trimming overlap that YouTube repeats."""
    existing = existing.strip()
    addition = addition.strip()
    if not addition:
        return existing
    if not existing:
        return addition
    # If the new fragment is already contained at the tail, skip it.
    if existing.endswith(addition):
        return existing
    # Trim the longest suffix of `existing` that is a prefix of `addition`.
    a_words = existing.split()
    b_words = addition.split()
    max_overlap = min(len(a_words), len(b_words))
    for k in range(max_overlap, 0, -1):
        if a_words[-k:] == b_words[:k]:
            b_words = b_words[k:]
            break
    if not b_words:
        return existing
    return existing + " " + " ".join(b_words)


def subtitle_excerpt(cues: list[dict[str, Any]], max_lines: int, max_chars: int) -> str:
    joined = " ".join(cue["text"] for cue in cues[:max_lines])
    return joined[:max_chars]


def language_instruction(output_language: str) -> str:
    """Build the response-language directive for analysis / frame summaries."""
    lang = (output_language or "auto").strip()
    if lang.lower() in ("", "auto", "source", "match"):
        return (
            "\n\nIMPORTANT: Write your entire response in the SAME language as the "
            "spoken content / transcript of this media. Do not translate it."
        )
    return (
        f"\n\nIMPORTANT: Write your entire response in {lang}, regardless of the "
        f"original language of the content."
    )


# ---------------------------------------------------------------------------
# Analysis / frame description / translation (all via Ollama)
# ---------------------------------------------------------------------------

def analyze_media(
    endpoint: str,
    model: str,
    media_path: Path,
    prompt: str,
    num_predict: int,
    keep_thinking: bool,
    frames: list[dict[str, Any]],
    subtitles: list[dict[str, Any]],
    analysis_frames: int,
    output_language: str = "auto",
) -> str:
    images: list[str] | None = None
    if frames and analysis_frames > 0:
        images = [encode_image(frame["path"]) for frame in frames[:analysis_frames]]

    context_parts: list[str] = []
    if images:
        context_parts.append(
            f"You are given {len(images)} sampled frames from the video as images."
        )
    if subtitles:
        excerpt = subtitle_excerpt(subtitles, max_lines=120, max_chars=6000)
        context_parts.append("Transcript excerpt:\n" + excerpt)

    text = prompt
    if context_parts:
        text = prompt + "\n\n" + "\n\n".join(context_parts)

    text += language_instruction(output_language)

    return ollama_chat(
        endpoint, model, text, images=images,
        num_predict=num_predict, keep_thinking=keep_thinking,
    )


def describe_frame(
    endpoint: str,
    model: str,
    frame_path: Path,
    keep_thinking: bool,
    output_language: str = "auto",
) -> str:
    prompt = (
        "Summarize the CONTENT of this moment in the video for study notes — this is a "
        "summary, not an image description. Read any slide title, bullet points, code, "
        "equations, numbers, or labels and summarize the idea or point being made in "
        "1-3 sentences. Do NOT describe the speaker's appearance, the room, the camera, "
        "or how the frame looks. "
        "If there is no readable content, just give a one-line summary of what this "
        "part is likely about."
    ) + language_instruction(output_language)
    return ollama_chat(
        endpoint, model, prompt,
        images=[encode_image(frame_path)],
        num_predict=256, keep_thinking=keep_thinking,
    )


def build_frame_breakdown(
    endpoint: str,
    model: str,
    video_path: Path,
    frames: list[dict[str, Any]],
    keep_thinking: bool,
    output_language: str = "auto",
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Describe each extracted frame with Gemma, write a Markdown breakdown, and
    return (markdown_path, notes) where each note is {image_path, time, description}."""
    if not frames:
        return None, []

    notes: list[dict[str, Any]] = []
    lines = [
        f"# Frame Breakdown - {video_path.stem}",
        "",
        f"- Source: `{video_path.resolve()}`",
        f"- Frames: {len(frames)}",
        "",
    ]
    for index, frame in enumerate(frames):
        frame_path: Path = frame["path"]
        timestamp = float(frame.get("time") or 0.0)
        try:
            description = describe_frame(
                endpoint, model, frame_path, keep_thinking, output_language
            )
        except Exception as exc:
            description = f"_Description failed: {exc}_"
        rel = frame_path.relative_to(video_path.parent).as_posix()
        lines.extend(
            [
                f"## {index + 1}. {timestamp:.1f}s",
                "",
                f"![Frame {index + 1}]({rel})",
                "",
                description,
                "",
            ]
        )
        notes.append(
            {
                "image_path": str(frame_path.resolve()),
                "time": timestamp,
                "description": description,
            }
        )

    breakdown_path = video_path.with_suffix(".frames.md")
    breakdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return breakdown_path, notes


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_study_markdown(
    path: Path,
    media_path: Path,
    analysis: str,
    subtitles: list[dict[str, Any]],
) -> None:
    lines = [
        "# Media Study Notes",
        "",
        f"- Media file: `{media_path.resolve()}`",
        "",
        "## Analysis",
        "",
        analysis.strip(),
        "",
        "## Subtitles",
        "",
    ]

    if not subtitles:
        lines.append("No subtitles were extracted.")
    else:
        for cue in subtitles:
            lines.extend(
                [
                    f"### {cue['index'] + 1}. {cue['start']:.2f}s - {cue['end']:.2f}s",
                    "",
                    cue.get("text", ""),
                    "",
                ]
            )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    endpoint = args.ollama_endpoint
    model = args.model

    local_media = args.local_media or args.local_video
    if local_media:
        video_path = Path(local_media).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Local media not found: {video_path}")
        print(f"Using local media: {video_path}")
    else:
        if not args.url:
            raise ValueError("YouTube URL is required unless --local-media is provided.")
        print("Downloading video...")
        video_path = download_youtube_video(args.url, output_dir, args.max_height)
        print(f"Downloaded: {video_path}")

    media_kind = media_kind_from_path(video_path)

    # Subtitles first: they drive the analysis context.
    subtitle_path = find_subtitle_file(video_path)
    thumbnail_path = find_thumbnail_file(video_path)
    subtitles: list[dict[str, Any]] = []
    if subtitle_path:
        print(f"Subtitle file: {subtitle_path}")
        subtitles = parse_subtitle(subtitle_path)
        raw_count = len(subtitles)
        subtitles = merge_subtitle_cues(subtitles)
        print(f"Parsed subtitles: {raw_count} -> merged into {len(subtitles)} scenes")
    else:
        # No subtitles (a local audio file, or a video without captions): transcribe
        # the speech. Whisper is the accurate primary path; Gemma audio is a fallback.
        print("No subtitle file found; transcribing speech with Whisper...")
        try:
            ollama_unload(endpoint, model)  # free VRAM for Whisper
            subtitles, detected = transcribe_with_whisper(video_path)
            print(f"Whisper transcribed {len(subtitles)} segments (language={detected})")
        except Exception as exc:
            print(f"Whisper unavailable ({exc}); falling back to Gemma audio...")
            try:
                subtitles = transcribe_audio_segments(
                    endpoint, model, video_path, args.keep_thinking
                )
                print(f"Audio-derived segments: {len(subtitles)}")
            except Exception as exc2:
                print(f"Audio understanding failed: {exc2}")

    # Extract frames once; reused for both the breakdown and the analysis images.
    frames: list[dict[str, Any]] = []
    if media_kind == "video" and args.frame_count > 0:
        print(f"Extracting {args.frame_count} frames...")
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames = extract_video_frames(video_path, args.frame_count, frames_dir)
        print(f"Extracted frames: {len(frames)}")

    print(f"Analyzing with Ollama model '{model}' at {endpoint}...")
    try:
        analysis = analyze_media(
            endpoint=endpoint,
            model=model,
            media_path=video_path,
            prompt=args.prompt,
            num_predict=args.max_new_tokens,
            keep_thinking=args.keep_thinking,
            frames=frames,
            subtitles=subtitles,
            analysis_frames=args.analysis_frames,
            output_language=args.output_language,
        )
    except Exception as exc:
        # Don't lose the whole run (subtitles, frames) if the analysis call
        # times out or errors — record the failure and keep going.
        print(f"Analysis call failed: {exc}")
        analysis = f"_Analysis failed: {exc}_"

    analysis_path = video_path.with_suffix(".analysis.md")
    analysis_path.write_text(analysis, encoding="utf-8")
    study_markdown_path = video_path.with_suffix(".study.md")

    # Subtitle translation was removed: it was the slowest step (one Ollama call
    # per chunk for the whole transcript). Subtitles are kept in their original
    # language for loop-practice and study.

    write_study_markdown(study_markdown_path, video_path, analysis, subtitles)

    frames_markdown_path: Path | None = None
    frame_notes: list[dict[str, Any]] = []
    if frames:
        print("Building frame breakdown...")
        try:
            frames_markdown_path, frame_notes = build_frame_breakdown(
                endpoint, model, video_path, frames, args.keep_thinking,
                args.output_language,
            )
            if frames_markdown_path:
                print(f"Saved frame breakdown: {frames_markdown_path}")
        except Exception as exc:
            print(f"Frame breakdown failed: {exc}")

    print("\n=== Analysis ===\n")
    print(analysis)
    print(f"\nSaved analysis: {analysis_path}")
    print(f"Saved study markdown: {study_markdown_path}")

    if args.json_summary_path:
        summary = {
            "video_path": str(video_path.resolve()),
            "media_kind": media_kind,
            "analysis_path": str(analysis_path.resolve()),
            "study_markdown_path": str(study_markdown_path.resolve()),
            "frames_markdown_path": str(frames_markdown_path.resolve())
            if frames_markdown_path
            else None,
            "frames": frame_notes,
            "subtitle_path": str(subtitle_path.resolve()) if subtitle_path else None,
            "thumbnail_path": str(thumbnail_path.resolve()) if thumbnail_path else None,
            "analysis": analysis,
            "subtitles": subtitles,
        }
        summary_path = Path(args.json_summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
