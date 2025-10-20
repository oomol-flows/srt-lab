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
    failed_batches = []
    translation_stats = {
        "total_subtitles": total_subtitles,
        "successful_translations": 0,
        "failed_translations": 0,
        "batches_processed": 0,
        "batches_failed": 0
    }

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

        # Create prompt for batch translation with clearer instructions
        source_text = "\n---\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(texts_to_translate)])

        if source_lang == "auto":
            prompt = f"""You are translating subtitles. Follow these instructions exactly:

1. Translate each subtitle text to {target_lang_name}.{style_instruction}
2. CRITICAL: You must respond with the EXACT same numbering format: [1], [2], [3], etc.
3. Each translation must be on a separate line
4. Do NOT add any explanations, notes, or extra text
5. Do NOT change the numbering
6. Do NOT skip any numbers - even if a subtitle is empty, special characters, or music notes, you MUST include it with its number
7. If a subtitle contains only symbols (like ♪, ..., --), translate or keep as appropriate

Example input:
[1] Hello world
[2] ♪
[3] How are you?

Example output:
[1] Translation of hello world
[2] ♪
[3] Translation of how are you

Now translate:
{source_text}

Your response:"""
        else:
            prompt = f"""You are translating subtitles. Follow these instructions exactly:

1. Translate each subtitle text from {source_lang_name} to {target_lang_name}.{style_instruction}
2. CRITICAL: You must respond with the EXACT same numbering format: [1], [2], [3], etc.
3. Each translation must be on a separate line
4. Do NOT add any explanations, notes, or extra text
5. Do NOT change the numbering
6. Do NOT skip any numbers - even if a subtitle is empty, special characters, or music notes, you MUST include it with its number
7. If a subtitle contains only symbols (like ♪, ..., --), translate or keep as appropriate

Example input:
[1] Hello world
[2] ♪
[3] How are you?

Example output:
[1] Translation of hello world
[2] ♪
[3] Translation of how are you

Now translate:
{source_text}

Your response:"""

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

        # Parse translated texts with robust error handling and multiple fallback strategies
        translated_lines = translated_text.strip().split("\n")
        # Filter out empty lines and clean up
        translated_lines = [line.strip() for line in translated_lines if line.strip()]
        logger.debug(f"Received {len(translated_lines)} translated lines for batch {batch_num}")
        logger.debug(f"Raw translation response: {repr(translated_text[:200])}...")

        # Create translation mapping with multiple strategies
        translation_map = {}
        numbered_translations = []
        unnumbered_translations = []

        # First pass: extract numbered translations with enhanced pattern matching
        for line in translated_lines:
            # Try to match various numbering patterns
            import re
            matched = False

            # Pattern 1: [1], [2], etc.
            numbered_match = re.match(r'^\[(\d+)\]\s*(.+)$', line)
            if numbered_match:
                num = int(numbered_match.group(1))
                text = numbered_match.group(2).strip()
                translation_map[num] = text
                numbered_translations.append((num, text))
                matched = True

            # Pattern 2: 1., 2., etc.
            if not matched:
                numbered_match = re.match(r'^(\d+)\.\s*(.+)$', line)
                if numbered_match:
                    num = int(numbered_match.group(1))
                    text = numbered_match.group(2).strip()
                    translation_map[num] = text
                    numbered_translations.append((num, text))
                    matched = True

            # Pattern 3: 1), 2), etc.
            if not matched:
                numbered_match = re.match(r'^(\d+)\)\s*(.+)$', line)
                if numbered_match:
                    num = int(numbered_match.group(1))
                    text = numbered_match.group(2).strip()
                    translation_map[num] = text
                    numbered_translations.append((num, text))
                    matched = True

            # Pattern 4: 1:, 2:, etc.
            if not matched:
                numbered_match = re.match(r'^(\d+):\s*(.+)$', line)
                if numbered_match:
                    num = int(numbered_match.group(1))
                    text = numbered_match.group(2).strip()
                    translation_map[num] = text
                    numbered_translations.append((num, text))
                    matched = True

            # No numbering, collect as potential fallback
            if not matched and line and not any(line.startswith(prefix) for prefix in ['[', '#', '-', '*']):
                unnumbered_translations.append(line)

        logger.debug(f"Found {len(numbered_translations)} numbered translations: {list(translation_map.keys())}")
        logger.debug(f"Found {len(unnumbered_translations)} unnumbered translations")

        # Track failed subtitles in this batch for retry
        batch_failed_indices = []

        # Process each subtitle with multiple fallback strategies
        for idx, (index, timestamp, original_text) in enumerate(batch):
            translated = None
            expected_num = idx + 1
            strategy_used = None

            # Strategy 1: Try exact numbering match
            if expected_num in translation_map:
                translated = translation_map[expected_num]
                strategy_used = "exact_match"
                logger.debug(f"Strategy 1 success: Found translation for subtitle {expected_num}")

            # Strategy 2: Try close numbering matches (LLM might have offset by 1)
            elif any(num in translation_map for num in [expected_num-1, expected_num+1]):
                closest_num = None
                min_diff = float('inf')
                for num in translation_map:
                    diff = abs(num - expected_num)
                    if diff < min_diff:
                        min_diff = diff
                        closest_num = num

                if min_diff <= 2:  # Allow up to 2 numbers difference
                    translated = translation_map[closest_num]
                    strategy_used = "close_match"
                    logger.warning(f"Strategy 2 used: Using translation from {closest_num} for subtitle {expected_num} (diff: {min_diff})")

            # Strategy 3: Use positional matching if we have unnumbered translations
            if translated is None and idx < len(unnumbered_translations) and unnumbered_translations[idx]:
                translated = unnumbered_translations[idx]
                strategy_used = "positional"
                logger.warning(f"Strategy 3 used: Using positional translation for subtitle {expected_num}")

            # Strategy 4: Try to extract from numbered translations by order
            if translated is None and numbered_translations and idx < len(numbered_translations):
                translated = numbered_translations[idx][1]
                strategy_used = "ordered"
                logger.warning(f"Strategy 4 used: Using ordered translation for subtitle {expected_num}")

            # Mark for retry if all strategies failed
            if translated is None:
                logger.error(f"All strategies failed for subtitle {expected_num}, marking for retry")
                batch_failed_indices.append(idx)
                translated = original_text  # Temporary fallback
                strategy_used = "original"

            # Clean up the translation
            translated = str(translated).strip()
            translated = translated.strip('"').strip("'").strip()
            if not translated:
                if strategy_used != "original":
                    batch_failed_indices.append(idx)
                translated = original_text
                logger.warning(f"Empty translation for subtitle {expected_num}, marking for retry")

            logger.debug(f"Final translation for {expected_num}: {translated[:50]}... (strategy: {strategy_used})")
            translated_subtitles.append((index, timestamp, translated))

        # Retry failed subtitles individually
        if batch_failed_indices:
            retry_failure_rate = len(batch_failed_indices) / len(batch)
            logger.warning(f"Batch {batch_num} has {len(batch_failed_indices)} failed subtitles ({retry_failure_rate*100:.1f}% failure rate)")

            # Only retry if failure rate is significant but not catastrophic
            if 0 < retry_failure_rate <= 0.5:  # Between 0% and 50% failure
                logger.info(f"Retrying {len(batch_failed_indices)} failed subtitles individually...")

                for fail_idx in batch_failed_indices:
                    subtitle_idx, subtitle_ts, original = batch[fail_idx]
                    global_idx = len(translated_subtitles) - len(batch) + fail_idx

                    retry_prompt = f"""Translate this single subtitle text to {target_lang_name}.{style_instruction}

Original text: {original.strip()}

Respond with ONLY the translation, no numbering, no explanations:"""

                    try:
                        retry_translation = call_llm_with_retry(
                            prompt=retry_prompt,
                            max_tokens=500,  # Single subtitle shouldn't need much
                            temperature=llm_config.get("temperature", 0),
                            top_p=llm_config.get("top_p", 0.5)
                        )

                        if retry_translation and retry_translation.strip() and retry_translation.strip() != original.strip():
                            # Clean up retry translation
                            retry_translation = retry_translation.strip().strip('"').strip("'").strip()
                            # Update the subtitle
                            translated_subtitles[global_idx] = (subtitle_idx, subtitle_ts, retry_translation)
                            logger.info(f"✓ Successfully retried subtitle at position {fail_idx+1}")
                        else:
                            logger.warning(f"✗ Retry failed for subtitle at position {fail_idx+1}, keeping original")

                    except Exception as e:
                        logger.error(f"✗ Retry exception for subtitle at position {fail_idx+1}: {str(e)}")

        # Verify translation completeness
        successful_translations = sum(1 for i, (_, _, orig) in enumerate(batch)
                                    if translated_subtitles[-len(batch)+i][2] != orig)
        failed_translations = len(batch) - successful_translations

        # Update statistics
        translation_stats["successful_translations"] += successful_translations
        translation_stats["failed_translations"] += failed_translations
        translation_stats["batches_processed"] += 1

        if failed_translations > 0:
            translation_stats["batches_failed"] += 1
            failed_batches.append({
                "batch_num": batch_num,
                "batch_size": len(batch),
                "failed_count": failed_translations,
                "success_rate": successful_translations / len(batch) * 100
            })

        logger.info(f"Batch {batch_num} translation summary: {successful_translations}/{len(batch)} subtitles translated successfully ({successful_translations/len(batch)*100:.1f}% success rate)")

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

        # Final translation statistics
        overall_success_rate = (translation_stats["successful_translations"] / translation_stats["total_subtitles"]) * 100
        logger.info("=" * 50)
        logger.info("TRANSLATION COMPLETION REPORT")
        logger.info("=" * 50)
        logger.info(f"✅ Total subtitles: {translation_stats['total_subtitles']}")
        logger.info(f"✅ Successfully translated: {translation_stats['successful_translations']}")
        logger.info(f"❌ Failed translations: {translation_stats['failed_translations']}")
        logger.info(f"📊 Overall success rate: {overall_success_rate:.1f}%")
        logger.info(f"📦 Batches processed: {translation_stats['batches_processed']}")
        logger.info(f"⚠️  Batches with issues: {translation_stats['batches_failed']}")
        logger.info(f"📁 Output file: {output_path}")

        if failed_batches:
            logger.warning("⚠️  Batch issues detected:")
            for batch_info in failed_batches:
                logger.warning(f"   Batch {batch_info['batch_num']}: {batch_info['failed_count']}/{batch_info['batch_size']} failed ({batch_info['success_rate']:.1f}% success)")

        if overall_success_rate >= 95:
            logger.info("🎉 Translation quality: EXCELLENT")
        elif overall_success_rate >= 85:
            logger.info("👍 Translation quality: GOOD")
        elif overall_success_rate >= 70:
            logger.warning("⚠️  Translation quality: ACCEPTABLE")
        else:
            logger.error("❌ Translation quality: NEEDS IMPROVEMENT")

        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Failed to save translated file: {str(e)}")
        raise

    return {
        "srt_file": str(output_path)
    }
