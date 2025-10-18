from oocana import Context
from faster_whisper import WhisperModel

#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio_file: str
    model_size: typing.Literal["tiny", "base", "small", "medium", "large-v3"]
    language: typing.Literal["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar", "pt", "it", "nl", "pl", "tr", "vi", "th", "id", "hi"]
    use_gpu: bool
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
    import os
    import sys
    from pathlib import Path
    import ctypes

    # Preload cuDNN libraries for faster-whisper GPU support
    cuda_lib_path = Path(sys.executable).parent.parent / "lib" / "python3.11" / "site-packages" / "nvidia" / "cudnn" / "lib"
    if cuda_lib_path.exists():
        try:
            # Preload cuDNN libraries in correct order
            ctypes.CDLL(str(cuda_lib_path / "libcudnn.so.9"))
            ctypes.CDLL(str(cuda_lib_path / "libcudnn_ops.so.9"))
            ctypes.CDLL(str(cuda_lib_path / "libcudnn_cnn.so.9"))
        except Exception as e:
            print(f"Warning: Failed to preload cuDNN libraries: {e}")

    audio_file_path = params["audio_file"]
    model_size = params["model_size"]
    language = params["language"]
    use_gpu = params["use_gpu"]

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
    # Configure device and compute type based on GPU availability
    import torch

    # Check CUDA availability first
    cuda_available = torch.cuda.is_available()

    if use_gpu and not cuda_available:
        print("=" * 60)
        print("⚠️  GPU requested but CUDA is not available")
        print("=" * 60)
        print("Falling back to CPU mode...")
        use_gpu = False

    if use_gpu:
        device = "cuda"
        compute_type = "float16"  # Better performance on GPU
        print("=" * 60)
        print("🚀 GPU Acceleration Mode Enabled")
        print("=" * 60)
        print(f"📌 Device: {device}")
        print(f"📌 Compute Type: {compute_type}")
        print(f"📌 Model Size: {model_size}")
        print("\nℹ️  Requirements for GPU:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - CUDA 12 installed")
        print("  - nvidia-cublas-cu12 and nvidia-cudnn-cu12 libraries")
        print("=" * 60)
    else:
        device = "cpu"
        compute_type = "int8"  # Optimized for CPU
        print("=" * 60)
        print("💻 CPU Mode")
        print("=" * 60)
        print(f"📌 Device: {device}")
        print(f"📌 Compute Type: {compute_type}")
        print(f"📌 Model Size: {model_size}")
        print("=" * 60)

    try:
        print("\n⏳ Initializing Whisper model...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if device == "cuda":
            print("✅ Successfully initialized model on GPU!")
            print("🎯 GPU acceleration is active")
        else:
            print("✅ Successfully initialized model on CPU")
        print("=" * 60 + "\n")
    except Exception as e:
        if device == "cuda":
            print(f"\n❌ GPU initialization failed!")
            print(f"Error: {str(e)}")
            print("\n⚠️  Possible reasons:")
            print("  1. No NVIDIA GPU available")
            print("  2. CUDA drivers not installed")
            print("  3. Missing CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12)")
            print("\n🔄 Falling back to CPU mode...")
            print("=" * 60 + "\n")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("✅ Successfully initialized model on CPU (fallback mode)")
            print("=" * 60 + "\n")
        else:
            raise

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
