from oocana import Context
from faster_whisper import WhisperModel

#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio_file: str
    model_size: typing.Literal["tiny", "base", "small", "medium", "large-v3"]
    language: typing.Literal["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar", "pt", "it", "nl", "pl", "tr", "vi", "th", "id", "hi"]
class Outputs(typing.TypedDict):
    srt_file: typing.NotRequired[str]
#endregion

def main(params: Inputs, context: Context) -> Outputs | None:
    """
    Convert audio file to SRT subtitle format using faster-whisper

    Args:
        params: Input parameters including audio file path, model size, and language
        context: OOMOL context object

    Returns:
        Dictionary with generated SRT file path
    """
    from pathlib import Path

    audio_file_path = params["audio_file"]
    model_size = params["model_size"]
    language = params["language"]

    # Language code mapping for faster-whisper
    lang_codes = {
        "auto": None,  # Let Whisper auto-detect
        "en": "en",
        "zh": "zh",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "ja": "ja",
        "ko": "ko",
        "ru": "ru",
        "ar": "ar",
        "pt": "pt",
        "it": "it",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "vi": "vi",
        "th": "th",
        "id": "id",
        "hi": "hi"
    }

    # Initialize faster-whisper model
    # Using CPU for compatibility, can be changed to "cuda" for GPU
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe audio file
    whisper_language = lang_codes.get(language)

    segments, info = model.transcribe(
        audio_file_path,
        language=whisper_language,
        vad_filter=True  # Enable voice activity detection for better segmentation
    )

    # Convert to SRT format
    srt_content = []
    audio_path = Path(audio_file_path)

    for idx, segment in enumerate(segments, start=1):
        # Format timestamp from seconds to SRT format (HH:MM:SS,mmm)
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)

        # SRT format:
        # 1
        # 00:00:00,000 --> 00:00:05,000
        # Subtitle text
        srt_content.append(f"{idx}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(segment.text.strip())
        srt_content.append("")  # Empty line between subtitles

    # Save SRT file
    output_filename = f"{audio_path.stem}.srt"
    output_path = audio_path.parent / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))

    return {
        "srt_file": str(output_path)
    }


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
