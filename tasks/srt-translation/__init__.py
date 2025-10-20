from oocana import Context
from openai import OpenAI
import time
import logging

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

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
        api_key=api_key,
        timeout=30.0
    )

    # Retry configuration
    max_retries = 3
    retry_delay = 1.0  # seconds

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

    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    def estimate_tokens(text):
        return len(text) // 4

    # Retry wrapper for API calls
    def call_llm_with_retry(prompt, max_tokens, temperature=0, top_p=0.5):
        """Call LLM with retry mechanism and proper error handling"""
        for attempt in range(max_retries):
            try:
                # Determine if we need streaming based on max_tokens
                use_streaming = max_tokens > 5000

                if use_streaming:
                    logger.info(f"Using streaming mode (max_tokens={max_tokens})")
                    # Streaming response for large token requests
                    response_stream = client.chat.completions.create(
                        model=llm_config["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stream=True
                    )

                    # Collect streamed content
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content

                    return full_response
                else:
                    # Regular response for smaller requests
                    response = client.chat.completions.create(
                        model=llm_config["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")

                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"All {max_retries} API call attempts failed")
                    raise

                # Check if we should retry based on error type
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["rate", "timeout", "connection"]):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # For non-retryable errors, fail immediately
                    raise

        return None

    # Calculate dynamic batch size with improved token management
    max_model_tokens = llm_config.get("max_tokens", 128000)

    # Conservative limits to ensure reliability
    # For streaming: we can use up to 80% of model context
    # For non-streaming: we must stay under 5000 tokens or use streaming
    safe_total_limit = int(max_model_tokens * 0.8)
    prompt_overhead = 800  # More conservative estimate for prompt overhead

    # Calculate safe input limits considering both input and output
    # Reserve equal space for input and output, plus prompt overhead
    max_input_tokens = (safe_total_limit - prompt_overhead) // 3  # 1/3 for input, 1/3 for output, 1/3 buffer

    # Ensure we don't exceed reasonable batch sizes
    max_input_tokens = min(max_input_tokens, 8000)  # Cap at 8k for stability

    logger.info(f"Token limits: max_model={max_model_tokens}, safe_total={safe_total_limit}, max_input={max_input_tokens}")

    # Translate subtitles in dynamic batches with progress tracking
    translated_subtitles = []
    total_subtitles = len(subtitles)
    processed_count = 0

    logger.info(f"Starting translation of {total_subtitles} subtitles")

    i = 0
    batch_num = 0
    while i < len(subtitles):
        batch = []
        current_tokens = 0

        # Build batch dynamically based on token count
        while i < len(subtitles):
            sub_text = subtitles[i][2].strip()
            sub_tokens = estimate_tokens(sub_text)

            # Check if adding this subtitle would exceed limit
            if current_tokens + sub_tokens > max_input_tokens and batch:
                break

            batch.append(subtitles[i])
            current_tokens += sub_tokens
            i += 1

            # Safety limit: don't batch more than 30 subtitles at once for better reliability
            if len(batch) >= 30:
                break

        if not batch:
            # Single subtitle is too large, process it anyway with a warning
            logger.warning(f"Processing large subtitle at position {i} individually")
            batch = [subtitles[i]]
            i += 1

        batch_num += 1
        logger.info(f"Processing batch {batch_num}: {len(batch)} subtitles, ~{current_tokens} tokens")

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

        # Call LLM for translation using retry mechanism
        safe_output_tokens = max_input_tokens  # Match input size for output

        try:
            translated_text = call_llm_with_retry(
                prompt=prompt,
                max_tokens=safe_output_tokens,
                temperature=llm_config.get("temperature", 0),
                top_p=llm_config.get("top_p", 0.5)
            )

            if not translated_text:
                logger.error(f"Failed to get translation for batch {batch_num}")
                # Use original text as fallback
                translated_text = source_text

        except Exception as e:
            logger.error(f"Translation failed for batch {batch_num}: {str(e)}")
            # Use original text as fallback
            translated_text = source_text

        # Parse translated texts with improved error handling
        translated_lines = translated_text.strip().split("\n")
        logger.debug(f"Received {len(translated_lines)} translated lines for batch {batch_num}")

        for idx, (index, timestamp, original_text) in enumerate(batch):
            # Find corresponding translated text
            translated = original_text  # fallback to original
            expected_prefix = f"[{idx+1}]"

            # Try to find the matching translated line
            for line in translated_lines:
                if line.strip().startswith(expected_prefix):
                    translated = line.replace(expected_prefix, "", 1).strip()
                    # Clean up any extra formatting
                    translated = translated.strip('"').strip("'").strip()
                    if not translated:
                        translated = original_text  # fallback if empty
                    break
            else:
                # If no match found, try alternative parsing
                logger.warning(f"Could not find translation for subtitle {idx+1} in batch {batch_num}, using original")
                # Try to match by position if numbering failed
                if idx < len(translated_lines):
                    line_text = translated_lines[idx].strip()
                    if not line_text.startswith('['):
                        translated = line_text
                        if not translated:
                            translated = original_text

            translated_subtitles.append((index, timestamp, translated))

        processed_count += len(batch)
        progress = (processed_count / total_subtitles) * 100
        logger.info(f"Progress: {processed_count}/{total_subtitles} ({progress:.1f}%)")

    # Generate output SRT content
    output_content = []
    for index, timestamp, text in translated_subtitles:
        output_content.append(f"{index}\n{timestamp}\n{text}\n")

    # Save translated SRT file
    input_path = Path(srt_file_path)
    output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
    output_path = input_path.parent / output_filename

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_content))

        logger.info(f"Translation completed successfully! Output saved to: {output_path}")
        logger.info(f"Translated {len(translated_subtitles)} subtitles in {batch_num} batches")

    except Exception as e:
        logger.error(f"Failed to save translated file: {str(e)}")
        raise

    return {
        "srt_file": str(output_path)
    }
