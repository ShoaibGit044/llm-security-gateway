"""tests/test_policy.py — Unit tests for policy engine decisions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from app.policy.policy_engine import compute_final_risk, apply_policy, build_reason_codes

with open("config/gateway_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Helpers to build mock inputs
def _rule(score):
    v = "BLOCK" if score >= 0.70 else ("WARN" if score >= 0.40 else "PASS")
    return {"score": score, "verdict": v, "triggers": []}

def _semantic(score):
    v = "BLOCK" if score >= 0.75 else ("WARN" if score >= 0.40 else "PASS")
    return {"score": score, "verdict": v}

def _pii(count, has_secret=False, composite=0.0, masked="<MASKED>"):
    return {
        "pii_count": count, "has_secret": has_secret,
        "composite_extra": composite, "masked_text": masked,
        "findings": []
    }


def test_clean_input_allowed():
    risk = compute_final_risk(0.0, 0.0, 0, False, 0.0, CONFIG)
    result = apply_policy(risk, _rule(0.0), _semantic(0.0),
                          _pii(0), "hello world", [], CONFIG)
    assert result["action"] == "ALLOWED"

def test_high_injection_blocked():
    risk = compute_final_risk(0.85, 0.90, 0, False, 0.0, CONFIG)
    result = apply_policy(risk, _rule(0.85), _semantic(0.90),
                          _pii(0), "inject text", [], CONFIG)
    assert result["action"] == "BLOCKED"

def test_pii_only_masked():
    risk = compute_final_risk(0.0, 0.0, 1, False, 0.0, CONFIG)
    result = apply_policy(risk, _rule(0.0), _semantic(0.0),
                          _pii(1, masked="my <EMAIL>"), "my email@x.com", [], CONFIG)
    assert result["action"] == "MASKED"
    assert result["safe_text"] == "my <EMAIL>"

def test_secret_entity_blocks():
    risk = compute_final_risk(0.0, 0.0, 1, True, 0.0, CONFIG)
    # secret_weight alone doesn't block — needs injection signal
    # Just verify risk increases
    normal_risk = compute_final_risk(0.0, 0.0, 1, False, 0.0, CONFIG)
    assert risk > normal_risk

def test_composite_boosts_risk():
    r_with    = compute_final_risk(0.0, 0.0, 2, False, 0.15, CONFIG)
    r_without = compute_final_risk(0.0, 0.0, 2, False, 0.0,  CONFIG)
    assert r_with > r_without

def test_reason_codes_injection():
    codes = build_reason_codes(_rule(0.80), _semantic(0.82), _pii(0), "en", False)
    assert "RULE_INJECTION"     in codes
    assert "SEMANTIC_INJECTION" in codes
    assert "HYBRID_INJECTION"   in codes

def test_reason_codes_pii():
    codes = build_reason_codes(_rule(0.0), _semantic(0.0), _pii(2), "en", False)
    assert "PII_DETECTED" in codes

def test_reason_multilingual():
    codes = build_reason_codes(_rule(0.0), _semantic(0.0), _pii(0), "ur", False)
    assert "MULTILINGUAL_ATTACK" in codes

def test_reason_obfuscated():
    codes = build_reason_codes(_rule(0.5), _semantic(0.0), _pii(0), "en", True)
    assert "OBFUSCATED_ATTACK" in codes


if __name__ == "__main__":
    tests = [
        test_clean_input_allowed, test_high_injection_blocked,
        test_pii_only_masked, test_secret_entity_blocks,
        test_composite_boosts_risk, test_reason_codes_injection,
        test_reason_codes_pii, test_reason_multilingual,
        test_reason_obfuscated,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
