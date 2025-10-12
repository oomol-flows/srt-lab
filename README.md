# Audio & Subtitle Translation Toolkit

A complete solution for converting audio files to subtitles and translating them into multiple languages - no coding required!

## What Does This Project Do?

This toolkit helps you work with audio and subtitle files in two main ways:

1. **Convert Audio to Subtitles** - Automatically transcribe speech from audio files into subtitle files (SRT format)
2. **Translate Subtitles** - Convert subtitles from one language to another while preserving timing and formatting

Perfect for content creators, educators, translators, and anyone working with multilingual video content.

---

## Available Blocks

### 1. Audio to SRT

**What it does:** Converts audio files into subtitle files with accurate timestamps.

**Use cases:**
- Creating subtitles for podcasts or videos
- Transcribing interviews or lectures
- Making content accessible with captions
- Extracting spoken content as text

**Features:**
- **Multiple audio formats supported**: MP3, WAV, M4A, FLAC, OGG, AAC
- **Multiple model sizes**: Choose between speed and accuracy
  - Tiny: Fastest processing, good for quick drafts
  - Base: Balanced speed and quality
  - Small: Better accuracy
  - Medium: High quality
  - Large-v3: Best accuracy, slower processing
- **18+ languages supported** including:
  - English, Chinese, Spanish, French, German
  - Japanese, Korean, Russian, Arabic, Portuguese
  - Italian, Dutch, Polish, Turkish, Vietnamese
  - Thai, Indonesian, Hindi
- **Auto-detect language**: Automatically identifies the spoken language

**How to use:**
1. Select your audio file
2. Choose a model size (Base recommended for most users)
3. Select the language or use auto-detect
4. Run the block to generate your SRT subtitle file

---

### 2. SRT Translation

**What it does:** Translates subtitle files from one language to another using advanced AI language models.

**Use cases:**
- Localizing video content for different markets
- Making educational content multilingual
- Creating foreign language versions of subtitles
- Adapting content for international audiences

**Features:**
- **18+ language pairs supported**: Same languages as Audio to SRT
- **Auto-detect source language**: Don't know the original language? Let AI figure it out
- **Translation styles**: Customize the tone and style
  - Formal: Professional, official tone
  - Casual: Conversational, relaxed
  - Professional: Business-appropriate
  - Slang: Informal, colloquial
  - Literary: Refined, artistic expression
- **Preserves timing**: Keeps all subtitle timestamps intact
- **AI-powered**: Uses advanced language models for natural translations

**How to use:**
1. Select your SRT subtitle file
2. Choose source language (or use auto-detect)
3. Select target language
4. (Optional) Pick a translation style
5. Configure AI model settings if needed
6. Run the block to generate translated subtitles

---

## Common Workflows

### Workflow 1: Audio to Multilingual Subtitles

Create subtitles in multiple languages from a single audio file:

```
Audio File → Audio to SRT → SRT Translation (Language 1) → English subtitles
                         → SRT Translation (Language 2) → Spanish subtitles
                         → SRT Translation (Language 3) → Chinese subtitles
```

### Workflow 2: Video Localization Pipeline

Convert your video audio into subtitles and translate:

```
Video → Extract Audio → Audio to SRT → SRT Translation → Localized Subtitles
```

### Workflow 3: Podcast Transcription & Translation

Transcribe podcasts and make them available in multiple languages:

```
Podcast MP3 → Audio to SRT → Original Language Subtitles
                           → SRT Translation → Translated Subtitles
```

---

## Getting Started

### Prerequisites

This is an OOMOL platform project. You'll need:
- OOMOL platform installed
- Internet connection (for AI model downloads on first use)
- Audio files or SRT files to process

### Installation

1. Import this project into your OOMOL platform
2. The system will automatically install required dependencies
3. Start building workflows with the available blocks

### First-Time Setup

When you first use the **Audio to SRT** block:
- The AI model will download automatically (this may take a few minutes)
- Subsequent uses will be much faster
- Larger models require more disk space but provide better accuracy

---

## Tips for Best Results

### Audio to SRT Tips

- **Use higher quality audio** for better transcription accuracy
- **Choose the right model size**:
  - Quick tests: Tiny or Base
  - Production use: Medium or Large-v3
- **Specify language when possible** instead of auto-detect for better results
- **Clear audio with minimal background noise** works best

### SRT Translation Tips

- **Use auto-detect** if you're unsure of the source language
- **Select appropriate translation style** based on your content:
  - Educational content: Formal or Professional
  - Entertainment: Casual
  - Marketing: Professional
  - Creative content: Literary
- **Review translations** for context-specific terms that may need manual adjustment
- **Adjust AI temperature settings** (in advanced options):
  - Lower temperature (0-0.3): More consistent, literal translations
  - Higher temperature (0.5-0.8): More creative, natural-sounding translations

---

## Technical Details

### Supported Audio Formats

- MP3 (MPEG Audio Layer 3)
- WAV (Waveform Audio File)
- M4A (MPEG-4 Audio)
- FLAC (Free Lossless Audio Codec)
- OGG (Ogg Vorbis)
- AAC (Advanced Audio Coding)

### Supported Languages

English, Chinese, Spanish, French, German, Japanese, Korean, Russian, Arabic, Portuguese, Italian, Dutch, Polish, Turkish, Vietnamese, Thai, Indonesian, Hindi

### Output Format

All subtitle files are generated in **SRT (SubRip)** format, which is widely compatible with:
- YouTube
- Vimeo
- Most video editing software
- Media players (VLC, Windows Media Player, etc.)

---

## Troubleshooting

### Audio to SRT Issues

**Problem:** Transcription is inaccurate
- **Solution**: Try a larger model size, specify the language explicitly, or use cleaner audio

**Problem:** Processing is very slow
- **Solution**: Use a smaller model (Tiny or Base) or check system resources

**Problem:** Language not detected correctly
- **Solution**: Manually select the language instead of using auto-detect

### SRT Translation Issues

**Problem:** Translation seems unnatural
- **Solution**: Try a different translation style or adjust the AI temperature

**Problem:** Technical terms translated incorrectly
- **Solution**: Use "Professional" style or manually review and edit specific terms

**Problem:** Formatting issues in output
- **Solution**: Check that your input SRT file is properly formatted

---

## Project Structure

```
srt-translation/
├── flows/                  # Workflow definitions
│   ├── flow-1/            # Example workflow
│   └── test-audio-to-srt/ # Audio transcription test workflow
├── tasks/                  # Reusable blocks
│   ├── audio-to-srt/      # Audio to subtitle conversion
│   └── srt-translation/   # Subtitle translation
├── package.oo.yaml         # OOMOL configuration
├── pyproject.toml          # Python dependencies
└── README.md               # This file
```

---

## FAQ

**Q: Can I use this offline?**
A: The Audio to SRT block works offline after initial model download. The SRT Translation block requires an internet connection to access AI models.

**Q: How long does transcription take?**
A: Depends on model size and audio length. As a rough guide:
- Tiny/Base: Near real-time
- Medium: 2-3x audio length
- Large-v3: 4-5x audio length

**Q: Is there a file size limit?**
A: No hard limit, but very large files may require more processing time and system resources.

**Q: Can I translate subtitles back to the original language?**
A: Yes, you can translate in any direction between supported languages.

**Q: What if my language isn't supported?**
A: The current version supports 18 major languages. Additional languages may be added in future updates.

---

## License & Credits

This project uses:
- **Faster-Whisper**: Speech recognition model
- **OOMOL Platform**: Visual workflow builder
- **LLM Integration**: AI-powered translation

---

## Support

For issues, questions, or feature requests:
1. Check the Troubleshooting section above
2. Review OOMOL platform documentation
3. Contact your OOMOL platform administrator

---

**Version:** 0.0.1
**Last Updated:** October 2025
