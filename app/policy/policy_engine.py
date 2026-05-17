"""
app/policy/policy_engine.py
Configurable policy engine — combines all risk signals into one decision.

HYBRID RISK FORMULA:
  final_risk = max(rule_score, semantic_score)
               + pii_weight      (if PII found)
               + secret_weight   (if API key found)
               + composite_extra (if composite entity detected)

  All weights are configurable in gateway_config.yaml.

DECISION TABLE:
  final_risk >= hybrid_block_threshold → BLOCK
  PII found + policy=mask             → MASK
  PII found + policy=block            → BLOCK
  Clean input                         → ALLOW

REASON CODES (new for lab final):
  Every decision includes a list of reason codes explaining WHY
  the decision was made. Makes the gateway auditable.

  Example: ["SEMANTIC_INJECTION", "PII_DETECTED", "COMPOSITE_ENTITY"]
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Reason code constants ─────────────────────────────────────
RC_RULE_INJECTION      = "RULE_INJECTION"
RC_SEMANTIC_INJECTION  = "SEMANTIC_INJECTION"
RC_HYBRID_INJECTION    = "HYBRID_INJECTION"
RC_PII_DETECTED        = "PII_DETECTED"
RC_SECRET_DETECTED     = "SECRET_DETECTED"
RC_COMPOSITE_ENTITY    = "COMPOSITE_ENTITY"
RC_MULTILINGUAL_ATTACK = "MULTILINGUAL_ATTACK"
RC_OBFUSCATED_ATTACK   = "OBFUSCATED_ATTACK"
RC_CLEAN               = "CLEAN_INPUT"


def compute_final_risk(
    rule_score:      float,
    semantic_score:  float,
    pii_count:       int,
    has_secret:      bool,
    composite_extra: float,
    config:          dict,
) -> float:
    """
    Compute the final combined risk score.

    Formula (from assignment):
      final_risk = max(rule_score, semantic_score)
                 + pii_weight      (if pii_count > 0)
                 + secret_weight   (if has_secret)
                 + composite_extra

    All weights loaded from gateway_config.yaml:
      pii_weight:    0.15  (default)
      secret_weight: 0.20  (default)

    DRY RUN:
      rule_score=0.63, semantic_score=0.88, pii_count=1, has_secret=False
      max(0.63, 0.88) = 0.88
      + 0.15 (pii found)
      = 1.00 (capped at 1.0)
      → BLOCK ✓
    """
    policy_cfg   = config.get("policy", {})
    pii_weight   = policy_cfg.get("pii_weight",    0.15)
    secret_weight= policy_cfg.get("secret_weight", 0.20)

    risk = max(rule_score, semantic_score)

    if pii_count > 0:
        risk += pii_weight
    if has_secret:
        risk += secret_weight
    risk += composite_extra

    return round(min(1.0, risk), 4)


def build_reason_codes(
    rule_result:      dict,
    semantic_result:  dict,
    pii_result:       dict,
    language:         str,
    is_obfuscated:    bool,
) -> list:
    """
    Build the list of human-readable reason codes for this decision.
    These explain WHY the gateway made its decision.
    """
    codes = []
    rule_thr = 0.40
    sem_thr  = 0.40

    if rule_result["score"] >= rule_thr:
        codes.append(RC_RULE_INJECTION)

    if semantic_result["score"] >= sem_thr:
        codes.append(RC_SEMANTIC_INJECTION)

    if rule_result["score"] >= rule_thr and semantic_result["score"] >= sem_thr:
        codes.append(RC_HYBRID_INJECTION)

    if pii_result["pii_count"] > 0:
        codes.append(RC_PII_DETECTED)

    if pii_result.get("has_secret"):
        codes.append(RC_SECRET_DETECTED)

    if pii_result.get("composite_extra", 0.0) > 0:
        codes.append(RC_COMPOSITE_ENTITY)

    if language not in ("en", "unknown"):
        codes.append(RC_MULTILINGUAL_ATTACK)

    if is_obfuscated:
        codes.append(RC_OBFUSCATED_ATTACK)

    if not codes:
        codes.append(RC_CLEAN)

    return codes


def apply_policy(
    final_risk:      float,
    rule_result:     dict,
    semantic_result: dict,
    pii_result:      dict,
    original_text:   str,
    reason_codes:    list,
    config:          dict,
) -> dict:
    """
    Make the final Allow / Mask / Block decision.

    Decision logic:
      1. If final_risk >= hybrid_block_threshold → BLOCK (injection/jailbreak)
      2. If pii_count > 0:
            policy=block → BLOCK
            policy=mask  → MASK (return masked text)
            policy=allow → ALLOW (log only)
      3. Otherwise → ALLOW (clean input)

    Returns:
      action    -- ALLOWED | MASKED | BLOCKED
      reason    -- human-readable explanation string
      safe_text -- output text (None if BLOCKED, masked if MASKED)
      reason_codes -- list of RC_* strings
    """
    block_thr = config.get("injection", {}).get("hybrid_block_threshold", 0.60)
    policy    = config.get("policy",    {}).get("mode", "mask")

    # Hard block ONLY if injection signal is present above threshold.
    # Pure PII prompts must NOT be blocked just because pii_weight
    # pushes final_risk over the limit — they should be MASKED.
    injection_risk = max(rule_result["score"], semantic_result["score"])
    if injection_risk >= block_thr:
        return {
            "action":       "BLOCKED",
            "reason":       (
                f"Injection/jailbreak detected "
                f"(injection_risk={injection_risk:.3f}, "
                f"rule={rule_result['score']}, "
                f"semantic={semantic_result['score']})"
            ),
            "safe_text":    None,
            "reason_codes": reason_codes,
        }

    # Clean input — no PII
    if pii_result["pii_count"] == 0:
        return {
            "action":       "ALLOWED",
            "reason":       "Clean input — no threats detected",
            "safe_text":    original_text,
            "reason_codes": reason_codes,
        }

    # PII present — apply configured policy
    if policy == "block":
        return {
            "action":       "BLOCKED",
            "reason":       (
                f"PII detected ({pii_result['pii_count']} entities) "
                "— policy=block"
            ),
            "safe_text":    None,
            "reason_codes": reason_codes,
        }
    elif policy == "mask":
        return {
            "action":       "MASKED",
            "reason":       (
                f"PII anonymized ({pii_result['pii_count']} entities) "
                "— policy=mask"
            ),
            "safe_text":    pii_result["masked_text"],
            "reason_codes": reason_codes,
        }
    else:  # allow
        return {
            "action":       "ALLOWED",
            "reason":       (
                f"PII detected but policy=allow "
                f"({pii_result['pii_count']} entities — logged only)"
            ),
            "safe_text":    original_text,
            "reason_codes": reason_codes,
        }
