from oocana import Context
from pathlib import Path
import re
import random

#region generated meta
import typing
class Inputs(typing.TypedDict):
    srt_file: str
    audio_file: str
    target_duration: float
    selection_method: typing.Literal["first", "longest", "densest", "random"]
class Outputs(typing.TypedDict):
    extracted_audio: typing.NotRequired[str]
    start_time: typing.NotRequired[float]
    end_time: typing.NotRequired[float]
    duration: typing.NotRequired[float]
#endregion


def parse_srt(srt_file: str) -> list[dict]:
    """
    Parse SRT file to extract subtitle segments with timestamps

    Args:
        srt_file: Path to SRT file

    Returns:
        List of subtitle segments with start_time, end_time, and text
    """
    segments = []

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Parse timestamp line (format: 00:00:00,000 --> 00:00:05,000)
        timestamp_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
            lines[1]
        )

        if timestamp_match:
            # Convert to seconds
            start_h, start_m, start_s, start_ms = map(int, timestamp_match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, timestamp_match.groups()[4:])

            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

            text = ' '.join(lines[2:])  # Combine all text lines

            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'text': text.strip()
            })

    return segments


def find_best_segment(segments: list[dict], target_duration: float, method: str) -> tuple[float, float]:
    """
    Find the best segment to extract based on the selection method

    Args:
        segments: List of parsed subtitle segments
        target_duration: Target duration in seconds
        method: Selection method (first, longest, densest, random)

    Returns:
        Tuple of (start_time, end_time) for the selected segment
    """
    if not segments:
        raise ValueError("No segments found in SRT file")

    # Tolerance: ±5 seconds from target duration, but minimum 10 seconds
    min_duration = max(10, target_duration - 5)
    max_duration = target_duration + 5

    # Group consecutive segments to form longer continuous speech blocks
    continuous_blocks = []
    current_block = [segments[0]]

    for i in range(1, len(segments)):
        # If gap between segments is less than 1 second, consider them continuous
        gap = segments[i]['start_time'] - current_block[-1]['end_time']
        if gap < 1.0:
            current_block.append(segments[i])
        else:
            continuous_blocks.append(current_block)
            current_block = [segments[i]]
    continuous_blocks.append(current_block)

    # Find blocks that match target duration
    valid_segments = []

    for block in continuous_blocks:
        block_start = block[0]['start_time']
        block_end = block[-1]['end_time']
        block_duration = block_end - block_start

        # If block is within acceptable range
        if min_duration <= block_duration <= max_duration:
            valid_segments.append({
                'start': block_start,
                'end': block_end,
                'duration': block_duration,
                'word_count': sum(len(seg['text'].split()) for seg in block),
                'segments': block
            })
        # If block is longer, try to extract a target-duration window
        elif block_duration > max_duration:
            # Slide window through the block
            for i in range(len(block)):
                window_start = block[i]['start_time']
                window_duration = 0
                window_segments = []

                for j in range(i, len(block)):
                    window_segments.append(block[j])
                    window_duration = block[j]['end_time'] - window_start

                    if min_duration <= window_duration <= max_duration:
                        valid_segments.append({
                            'start': window_start,
                            'end': block[j]['end_time'],
                            'duration': window_duration,
                            'word_count': sum(len(seg['text'].split()) for seg in window_segments),
                            'segments': window_segments
                        })
                    elif window_duration > max_duration:
                        break

    if not valid_segments:
        # Fallback: if no valid segments found, use the entire audio or the longest block
        all_candidates = []
        for block in continuous_blocks:
            block_start = block[0]['start_time']
            block_end = block[-1]['end_time']
            block_duration = block_end - block_start
            all_candidates.append({
                'start': block_start,
                'end': block_end,
                'duration': block_duration,
                'word_count': sum(len(seg['text'].split()) for seg in block),
                'segments': block
            })

        if not all_candidates:
            raise ValueError("No segments found in SRT file")

        # Use the longest or densest block available
        if method == "densest":
            valid_segments = [max(all_candidates, key=lambda x: x['word_count'])]
        else:
            valid_segments = [max(all_candidates, key=lambda x: x['duration'])]

    # Select based on method
    if method == "first":
        selected = valid_segments[0]
    elif method == "longest":
        selected = max(valid_segments, key=lambda x: x['duration'])
    elif method == "densest":
        selected = max(valid_segments, key=lambda x: x['word_count'])
    elif method == "random":
        selected = random.choice(valid_segments)
    else:
        selected = valid_segments[0]

    return selected['start'], selected['end']


def extract_audio_segment(audio_file: str, start_time: float, end_time: float, output_path: str):
    """
    Extract audio segment using ffmpeg directly

    Args:
        audio_file: Path to input audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save extracted audio
    """
    import subprocess

    duration = end_time - start_time

    # Use ffmpeg to extract audio segment with proper encoding
    cmd = [
        'ffmpeg',
        '-i', audio_file,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # Copy codec for fast extraction
        '-y',  # Overwrite output file
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def get_audio_duration(audio_file: str) -> float:
    """Get audio duration using ffprobe"""
    import subprocess
    import json

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        audio_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return float(data['format']['duration'])


def main(params: Inputs, context: Context) -> Outputs:
    """
    Extract a ~25-second audio segment based on Whisper SRT timestamps

    Args:
        params: Input parameters
        context: OOMOL context object

    Returns:
        Dictionary with extracted audio file path and segment information
    """
    srt_file = params["srt_file"]
    audio_file = params["audio_file"]
    target_duration = params["target_duration"]
    selection_method = params["selection_method"]

    # Get audio duration
    audio_duration = get_audio_duration(audio_file)

    # Parse SRT file
    segments = parse_srt(srt_file)

    if not segments:
        raise ValueError("No valid segments found in SRT file")

    # Find best segment
    start_time, end_time = find_best_segment(segments, target_duration, selection_method)

    # Ensure segment is within audio bounds
    if end_time > audio_duration:
        end_time = audio_duration
        if start_time >= end_time:
            start_time = max(0, end_time - target_duration)

    duration = end_time - start_time

    # Generate output filename
    audio_path = Path(audio_file)
    output_filename = f"{audio_path.stem}_extracted_{int(start_time)}s-{int(end_time)}s{audio_path.suffix}"
    output_path = audio_path.parent / output_filename

    # Extract audio segment
    extract_audio_segment(audio_file, start_time, end_time, str(output_path))

    return {
        "extracted_audio": str(output_path),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration
    }
