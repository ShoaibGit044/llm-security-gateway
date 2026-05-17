"""tests/test_pii.py — Unit tests for Presidio PII detection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from app.pii.presidio_custom import scan_pii

with open("config/gateway_config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def test_email_detected():
    r = scan_pii("My email is shoaib@ciit.edu.pk please contact me.", CONFIG)
    types = [f["entity_type"] for f in r["findings"]]
    assert "EMAIL_ADDRESS" in types, f"EMAIL not found. Got: {types}"

def test_student_id_detected():
    r = scan_pii("My registration number is FA24-BCS-044.", CONFIG)
    types = [f["entity_type"] for f in r["findings"]]
    assert "STUDENT_ID" in types, f"STUDENT_ID not found. Got: {types}"

def test_cnic_detected():
    r = scan_pii("My CNIC is 37405-1234567-9 for verification.", CONFIG)
    types = [f["entity_type"] for f in r["findings"]]
    assert "PK_CNIC" in types, f"PK_CNIC not found. Got: {types}"

def test_api_key_detected():
    r = scan_pii("My key is sk-abcdefghijklmnopqrstuvwxyz12345678.", CONFIG)
    types = [f["entity_type"] for f in r["findings"]]
    assert "API_KEY" in types, f"API_KEY not found. Got: {types}"
    assert r["has_secret"] is True

def test_email_masked():
    r = scan_pii("Send report to admin@example.com", CONFIG)
    assert "<EMAIL_ADDRESS>" in r["masked_text"], "Email not masked"
    assert "admin@example.com" not in r["masked_text"]

def test_composite_student_email():
    r = scan_pii("Student FA24-BCS-044 email shoaib@ciit.edu.pk.", CONFIG)
    assert r["composite_extra"] > 0, "Composite boost expected"

def test_clean_no_pii():
    r = scan_pii("What is machine learning?", CONFIG)
    assert r["pii_count"] == 0, f"Expected 0 PII, got {r['pii_count']}"

def test_overlap_resolution():
    # CNIC could overlap with phone pattern — check only one is returned
    r = scan_pii("CNIC: 37405-1234567-9", CONFIG)
    spans = [(f["start"], f["end"]) for f in r["findings"]]
    for i, (s1, e1) in enumerate(spans):
        for j, (s2, e2) in enumerate(spans):
            if i != j:
                assert not (s1 < e2 and s2 < e1), "Overlapping spans detected!"


if __name__ == "__main__":
    tests = [
        test_email_detected, test_student_id_detected,
        test_cnic_detected, test_api_key_detected,
        test_email_masked, test_composite_student_email,
        test_clean_no_pii, test_overlap_resolution,
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
