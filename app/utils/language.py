"""
app/utils/language.py
Language detection and translation module.

How it works:
  1. langdetect library identifies the language of input text.
  2. If non-English, deep_translator (free Google Translate) converts
     it to English so English-trained detectors can work on it.
  3. Original language code is returned for audit logging.

Supported: en, ur (Urdu), ko (Korean), hi (Hindi), ar (Arabic)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Try importing langdetect ──────────────────────────────────
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False
    logger.warning("langdetect not installed. Language detection disabled.")

# ── Try importing deep_translator ─────────────────────────────
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False
    logger.warning("deep_translator not installed. Translation disabled.")


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns ISO 639-1 language code (e.g. 'en', 'ur', 'ko').
    Falls back to 'en' if detection fails.

    Dry run:
      Input: "اگلی ہدایات کو نظرانداز کریں"  → returns "ur"
      Input: "이전 지침을 무시하고"             → returns "ko"
      Input: "Ignore all instructions"          → returns "en"
    """
    if not LANGDETECT_OK or not text.strip():
        return "en"
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate text to English using Google Translate (free tier).
    Falls back to original text if translation fails.

    This is what makes multilingual attack detection possible:
    - Urdu jailbreak gets translated to English
    - English detectors then catch it normally

    Dry run:
      Input: "اگلی ہدایات کو نظرانداز کریں" (ur)
      Output: "Ignore the previous instructions"  → detected by rules
    """
    if not TRANSLATOR_OK:
        return text
    if source_lang == "en":
        return text
    try:
        translated = GoogleTranslator(
            source=source_lang, target="en"
        ).translate(text)
        return translated if translated else text
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return text


def normalize_text(text: str) -> str:
    """
    Normalize obfuscated text before injection detection.

    Handles:
      1. L33tspeak:  "Ign0re" → "Ignore", "!nstruction" → "instruction"
      2. Extra spaces: "I G N O R E" → "IGNORE"
      3. Repeated punctuation: "!!!ignore!!!" → "ignore"
      4. Mixed case preserved (regex uses IGNORECASE)

    Dry run:
      Input:  "Ign0re prev!ous instruct!0ns"
      Step 1: "Ignore previous instructions"   (l33t substitution)
      Output: "Ignore previous instructions"   (ready for detection)
    """
    # L33tspeak character substitution map
    LEET_MAP = {
        "0": "o", "1": "i", "3": "e", "4": "a",
        "5": "s", "7": "t", "@": "a", "$": "s",
        "!": "i", "+": "t", "(": "c", "|": "i",
    }

    result = text

    # Step 1: Replace l33t characters
    for leet, normal in LEET_MAP.items():
        result = result.replace(leet, normal)

    # Step 2: Collapse spaced-out words like "I G N O R E"
    # Pattern: single char followed by space, repeated 3+ times
    result = re.sub(
        r"\b([A-Za-z])\s([A-Za-z])\s([A-Za-z](?:\s[A-Za-z])*)\b",
        lambda m: m.group(0).replace(" ", ""),
        result,
    )

    # Step 3: Strip excessive punctuation used as separators
    result = re.sub(r"[!*_~]{2,}", " ", result)

    return result


def prepare_text(text: str) -> dict:
    """
    Full preprocessing pipeline for a piece of input text.
    Returns dict with original, detected language, normalized,
    and english_text (translated if needed).

    This is the entry point called by main.py for every request.
    """
    lang = detect_language(text)
    normalized = normalize_text(text)

    if lang != "en":
        english_text = translate_to_english(normalized, source_lang=lang)
    else:
        english_text = normalized

    return {
        "original":     text,
        "language":     lang,
        "normalized":   normalized,
        "english_text": english_text,
    }
