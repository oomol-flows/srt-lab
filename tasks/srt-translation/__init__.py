from oocana import Context
from openai import OpenAI

#region generated meta
import typing
from oocana import LLMModelOptions
class Inputs(typing.TypedDict):
    srt_file: str
    source_language: typing.Literal["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar", "pt", "it", "nl", "pl", "tr", "vi", "th", "id", "hi"]
    target_language: typing.Literal["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar", "pt", "it", "nl", "pl", "tr", "vi", "th", "id", "hi"]
    translation_style: typing.Literal["formal", "casual", "professional", "slang", "literary"] | None
    llm: LLMModelOptions
class Outputs(typing.TypedDict):
    srt_file: typing.NotRequired[str]
#endregion

def main(params: Inputs, context: Context) -> Outputs | None:
    """
    Translate SRT subtitle file using LLM via OpenAI SDK

    Args:
        params: Input parameters including SRT file path, source/target languages, and LLM config
        context: OOMOL context object

    Returns:
        Dictionary with translated SRT file path
    """
    import re
    from pathlib import Path

    srt_file_path = params["srt_file"]
    source_lang = params["source_language"]
    target_lang = params["target_language"]
    translation_style = params.get("translation_style")
    llm_config = params["llm"]

    # Language code to full name mapping
    lang_names = {
        "auto": "Auto Detect",
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "ar": "Arabic",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "hi": "Hindi"
    }

    # Initialize OpenAI client with OOMOL environment
    base_url = context.oomol_llm_env.get("base_url_v1")
    api_key = context.oomol_llm_env.get("api_key")

    if not base_url or not api_key:
        raise ValueError("Missing LLM configuration: base_url_v1 or api_key not found in oomol_llm_env")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Read SRT file
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT file into subtitle blocks
    subtitle_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n?)+?)(?=\n\d+\n|\Z)'
    subtitles = re.findall(subtitle_pattern, srt_content, re.MULTILINE)

    if not subtitles:
        raise ValueError("No valid subtitles found in SRT file")

    # Prepare translation prompt
    source_lang_name = lang_names.get(source_lang, source_lang)
    target_lang_name = lang_names.get(target_lang, target_lang)

    # Style mapping
    style_descriptions = {
        "formal": "in a formal and polite tone",
        "casual": "in a casual and conversational tone",
        "professional": "in a professional and technical tone",
        "slang": "using slang and colloquial expressions",
        "literary": "in a literary and elegant style"
    }
    style_instruction = ""
    if translation_style and translation_style in style_descriptions:
        style_instruction = f" Use {style_descriptions[translation_style]}."

    # Translate subtitles in batches
    translated_subtitles = []
    batch_size = 10  # Process 10 subtitles at a time

    for i in range(0, len(subtitles), batch_size):
        batch = subtitles[i:i+batch_size]
        texts_to_translate = [sub[2].strip() for sub in batch]

        # Create prompt for batch translation
        source_text = "\n---\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(texts_to_translate)])

        if source_lang == "auto":
            prompt = f"""Translate the following subtitle texts to {target_lang_name}.{style_instruction}
Maintain the numbering format [1], [2], etc. in your response.
Only provide the translated texts, one per line with the same numbering.

{source_text}"""
        else:
            prompt = f"""Translate the following subtitle texts from {source_lang_name} to {target_lang_name}.{style_instruction}
Maintain the numbering format [1], [2], etc. in your response.
Only provide the translated texts, one per line with the same numbering.

{source_text}"""

        # Call LLM for translation using OpenAI SDK
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0),
            top_p=llm_config.get("top_p", 0.5),
            max_tokens=llm_config.get("max_tokens", 4096)
        )

        translated_text = response.choices[0].message.content

        # Parse translated texts
        translated_lines = translated_text.strip().split("\n")

        for idx, (index, timestamp, original_text) in enumerate(batch):
            # Find corresponding translated text
            translated = original_text  # fallback to original

            for line in translated_lines:
                if line.startswith(f"[{idx+1}]"):
                    translated = line.replace(f"[{idx+1}]", "").strip()
                    break

            translated_subtitles.append((index, timestamp, translated))

    # Generate output SRT content
    output_content = []
    for index, timestamp, text in translated_subtitles:
        output_content.append(f"{index}\n{timestamp}\n{text}\n")

    # Save translated SRT file
    input_path = Path(srt_file_path)
    output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
    output_path = input_path.parent / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))

    return {
        "srt_file": str(output_path)
    }
