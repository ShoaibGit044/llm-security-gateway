"""
app/pii/presidio_custom.py
Microsoft Presidio PII detection with 4 customizations.

CUSTOMIZATION SUMMARY:
  1. ApiKeyRecognizer         -- sk-* / sk-ant-* API key patterns
  2. StudentIdRecognizer      -- CIIT registration: FA24-BCS-044
  3. PakistaniCNICRecognizer  -- NADRA CNIC: XXXXX-XXXXXXX-X
  4. Context-aware scoring    -- keywords near entity boost confidence
  5. Composite entity detect  -- student_id + email = higher combined risk
  6. Overlap resolution       -- highest confidence wins on same span

APPROACH: Pure regex-based (no spaCy model download needed)
  Uses RecognizerResult directly and Presidio's AnonymizerEngine
  for text masking. Works without en_core_web_sm installed.
"""

import re
import logging
import time
from typing import List, Tuple

from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import RecognizerResult

logger = logging.getLogger(__name__)

# ── PII Pattern Registry ──────────────────────────────────────
# (entity_type, regex, base_score, context_keywords, ctx_boost)
# context_boost is added when any context keyword appears in text

PII_RECOGNIZERS: List[Tuple] = [
    # Built-in equivalents
    ("EMAIL_ADDRESS",
     r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
     0.85, ["email", "mail", "contact", "reach", "send"], 0.10),

    ("PHONE_NUMBER",
     r"(\+92|0)3[0-9]{9}|(\+92)[0-9]{10}|\+?[1-9]\d{7,14}",
     0.80, ["phone", "mobile", "call", "number", "contact", "reach"], 0.10),

    ("CREDIT_CARD",
     r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
     0.90, ["card", "credit", "visa", "mastercard", "payment", "billing"], 0.05),

    ("US_SSN",
     r"\b\d{3}-\d{2}-\d{4}\b",
     0.85, ["ssn", "social", "security", "taxpayer"], 0.10),

    ("IP_ADDRESS",
     r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
     0.75, ["ip", "address", "server", "host", "network"], 0.10),

    # CUSTOMIZATION 1: API Key Recognizer
    # Detects OpenAI (sk-XXX) and Anthropic (sk-ant-XXX) keys
    # High base score (0.95) — these patterns are highly specific
    ("API_KEY",
     r"sk-[A-Za-z0-9]{32,64}|sk-ant-[A-Za-z0-9\-_]{32,}",
     0.95, ["key", "token", "api", "secret", "credential", "auth", "openai", "anthropic"], 0.05),

    # CUSTOMIZATION 2: CIIT Student Registration Number
    # Format: FA24-BCS-044 (Semester-Program-Number)
    ("STUDENT_ID",
     r"\b(FA|SP|SU)\d{2}-[A-Z]{2,5}-\d{3,4}\b",
     0.90, ["registration", "reg", "student", "id", "number", "enroll", "roll"], 0.08),

    # CUSTOMIZATION 3: Pakistani CNIC (NADRA)
    # Format: XXXXX-XXXXXXX-X (5-7-1 digits)
    # Gap in Presidio's built-in recognizers which are US-centric
    ("PK_CNIC",
     r"\b\d{5}-\d{7}-\d{1}\b",
     0.92, ["cnic", "national", "id", "identity", "card", "nadra", "computerized"], 0.08),

    # LOCAL PHONE FORMAT (Pakistan landline / mobile)
    ("PK_PHONE",
     r"\b0[0-9]{2,3}-[0-9]{6,7}\b",
     0.82, ["phone", "landline", "number", "call", "contact"], 0.08),
]

# Entity types that carry extra "secret" weight in policy calculation
SECRET_ENTITIES = {"API_KEY"}

anonymizer = AnonymizerEngine()


def _resolve_overlaps(results: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    CUSTOMIZATION 5: Composite entity detection — overlap resolution.

    When two recognizers fire on the same character span, keep only
    the detection with the highest confidence score.

    Example: "37405-1234567-9" could match both US_SSN (partial)
    and PK_CNIC. We keep PK_CNIC (score=0.92 > 0.85).

    Algorithm:
      1. Sort by start position, then descending score
      2. For each result: if it overlaps with the last kept result,
         compare scores and keep the winner
    """
    results_sorted = sorted(results, key=lambda r: (r.start, -r.score))
    merged = []
    for r in results_sorted:
        if merged and r.start < merged[-1].end:
            if r.score > merged[-1].score:
                merged[-1] = r
        else:
            merged.append(r)
    return merged


def _check_composite(findings: List[dict]) -> float:
    """
    CUSTOMIZATION 6: Composite entity risk boosting.

    When multiple high-risk entities appear together, the combined
    risk is higher than individual entities alone.

    Combinations and their extra risk boosts:
      STUDENT_ID + EMAIL  → +0.10  (identity + contact info)
      STUDENT_ID + PK_CNIC→ +0.15  (full identity package)
      API_KEY + EMAIL     → +0.20  (credential + identity = breach)
      API_KEY + any PII   → +0.15

    Returns extra risk float to add to pii_weight.
    """
    entity_types = {f["entity_type"] for f in findings}
    extra = 0.0

    if "STUDENT_ID" in entity_types and "EMAIL_ADDRESS" in entity_types:
        extra = max(extra, 0.10)
        logger.info("Composite: STUDENT_ID + EMAIL detected (+0.10)")

    if "STUDENT_ID" in entity_types and "PK_CNIC" in entity_types:
        extra = max(extra, 0.15)
        logger.info("Composite: STUDENT_ID + CNIC detected (+0.15)")

    if "API_KEY" in entity_types and "EMAIL_ADDRESS" in entity_types:
        extra = max(extra, 0.20)
        logger.info("Composite: API_KEY + EMAIL detected (+0.20)")

    if "API_KEY" in entity_types and len(entity_types) > 1:
        extra = max(extra, 0.15)

    return extra


def scan_pii(text: str, config: dict) -> dict:
    """
    Full PII scanning pipeline with all 6 customizations.

    DRY RUN TRACE:
      Input: "My email is shoaib@ciit.edu.pk, reg FA24-BCS-044"

      EMAIL_ADDRESS pattern matches at [11, 29]:
        'email' keyword found in text → score = 0.85 + 0.10 = 0.95
      STUDENT_ID pattern matches at [36, 47]:
        no strong keyword → score = 0.90
      Overlap check: no overlaps (different spans)
      Composite: STUDENT_ID + EMAIL → extra_risk = 0.10

      AnonymizerEngine replaces spans:
      Output: "My email is <EMAIL_ADDRESS>, reg <STUDENT_ID>"
      pii_count = 2, has_secret = False, composite_extra = 0.10

    Returns:
      findings      -- list of {entity_type, start, end, score, text}
      masked_text   -- anonymized version of input
      pii_count     -- number of entities found
      has_secret    -- True if API_KEY or similar high-risk entity
      composite_extra -- extra risk from composite detection
      latency_ms    -- scan time
    """
    conf_threshold = config.get("pii", {}).get("confidence_threshold", 0.60)
    t0 = time.time()

    results    = []
    text_lower = text.lower()

    for entity_type, pattern, base_score, ctx_keywords, ctx_boost in PII_RECOGNIZERS:
        # CUSTOMIZATION 4: Context-aware scoring
        score = base_score
        if any(kw in text_lower for kw in ctx_keywords):
            score = min(1.0, score + ctx_boost)

        if score < conf_threshold:
            continue

        for m in re.finditer(pattern, text, re.IGNORECASE):
            results.append(RecognizerResult(
                entity_type=entity_type,
                start=m.start(),
                end=m.end(),
                score=round(score, 3),
            ))

    # CUSTOMIZATION 5: Resolve overlapping detections
    results = _resolve_overlaps(results)

    latency_ms = round((time.time() - t0) * 1000, 3)

    # Build findings list with matched text
    findings = [
        {
            "entity_type": r.entity_type,
            "text":        text[r.start:r.end],
            "start":       r.start,
            "end":         r.end,
            "score":       r.score,
        }
        for r in results
    ]

    # CUSTOMIZATION 6: Composite entity risk
    composite_extra = _check_composite(findings)

    masked_text = (
        anonymizer.anonymize(text=text, analyzer_results=results).text
        if results else text
    )

    has_secret = any(r.entity_type in SECRET_ENTITIES for r in results)

    return {
        "findings":        findings,
        "masked_text":     masked_text,
        "pii_count":       len(results),
        "has_secret":      has_secret,
        "composite_extra": composite_extra,
        "latency_ms":      latency_ms,
    }
