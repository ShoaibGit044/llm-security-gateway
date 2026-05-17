"""
run_evaluation.py
Reproducible evaluation script for the LLM Security Gateway.

Usage:
  python run_evaluation.py

Output files:
  results/evaluation_results.csv  -- per-row results
  results/metrics_summary.json    -- accuracy, precision, recall, F1
  results/audit_log.jsonl         -- updated audit log

This script runs every row in data/final_eval.csv through the
full gateway pipeline and computes all metrics required by the
assignment report tables.
"""

import csv
import json
import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)  # suppress info during eval

import yaml
from app.detectors.rule_detector     import detect_injection_rules
from app.detectors.semantic_detector import detect_injection_semantic, get_semantic_detector
from app.pii.presidio_custom         import scan_pii
from app.policy.policy_engine        import (
    compute_final_risk, build_reason_codes, apply_policy
)
from app.utils.language import prepare_text

# ── Load config ───────────────────────────────────────────────
with open("config/gateway_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

os.makedirs("results", exist_ok=True)

# ── Load semantic detector (trains once) ──────────────────────
print("Loading models...")
semantic_detector = get_semantic_detector()
print("Models ready.\n")


def run_single(text: str, row_id: str) -> dict:
    """Run full pipeline on one text, return result dict."""
    t0 = time.time()
    prep           = prepare_text(text)
    lang           = prep["language"]
    english_text   = prep["english_text"]
    normalized     = prep["normalized"]
    is_obfuscated  = normalized.lower() != text.lower()

    rule_result     = detect_injection_rules(english_text, CONFIG)
    semantic_result = detect_injection_semantic(english_text, CONFIG, semantic_detector)
    pii_result      = scan_pii(text, CONFIG)

    final_risk = compute_final_risk(
        rule_score      = rule_result["score"],
        semantic_score  = semantic_result["score"],
        pii_count       = pii_result["pii_count"],
        has_secret      = pii_result["has_secret"],
        composite_extra = pii_result["composite_extra"],
        config          = CONFIG,
    )
    reason_codes = build_reason_codes(
        rule_result, semantic_result, pii_result, lang, is_obfuscated
    )
    policy_result = apply_policy(
        final_risk, rule_result, semantic_result,
        pii_result, text, reason_codes, CONFIG
    )
    elapsed = round((time.time() - t0) * 1000, 3)

    return {
        "language":       lang,
        "rule_score":     rule_result["score"],
        "semantic_score": semantic_result["score"],
        "final_risk":     final_risk,
        "pii_count":      pii_result["pii_count"],
        "decision":       policy_result["action"],
        "reason_codes":   "|".join(reason_codes),
        "latency_ms":     elapsed,
    }


# ── Run evaluation ────────────────────────────────────────────
print("=" * 68)
print(" LLM SECURITY GATEWAY — FULL EVALUATION")
print(f" FA24-BCS-044 | Muhammad Shoaib Ahmed | CSC 262 Lab Final")
print("=" * 68)

eval_rows = []
with open("data/final_eval.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows   = list(reader)

results = []
for i, row in enumerate(rows):
    res = run_single(row["prompt"], row["id"])
    expected = row["expected_policy"].strip().upper()
    got      = res["decision"].upper()

    # Normalise: MASK == MASKED, ALLOW == ALLOWED, BLOCK == BLOCKED
    exp_norm = expected.rstrip("ED")  # ALLOWED→ALLOW, MASKED→MASK, BLOCKED→BLOCK
    got_norm = got.rstrip("ED")
    correct  = exp_norm == got_norm or expected == got

    results.append({
        "id":             row["id"],
        "prompt":         row["prompt"][:60],
        "language":       res["language"],
        "attack_type":    row["attack_type"],
        "has_pii":        row["has_pii"],
        "expected":       row["expected_policy"],
        "got":            res["decision"],
        "correct":        str(correct),
        "rule_score":     res["rule_score"],
        "semantic_score": res["semantic_score"],
        "final_risk":     res["final_risk"],
        "pii_count":      res["pii_count"],
        "reason_codes":   res["reason_codes"],
        "latency_ms":     res["latency_ms"],
    })

    mark = "PASS" if correct else "FAIL"
    if (i+1) % 10 == 0 or not correct:
        print(f"  [{mark}] {row['id']} | {row['attack_type'][:20]:<20} | "
              f"exp={row['expected_policy']:<8} got={res['decision']:<8} | "
              f"risk={res['final_risk']:.3f} | {res['latency_ms']:.1f}ms")

# ── Write results CSV ─────────────────────────────────────────
with open("results/evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ── Compute metrics ───────────────────────────────────────────
correct_total = sum(1 for r in results if r["correct"] == "True")
total         = len(results)
accuracy      = round(correct_total / total * 100, 2)

# True Positive = attack correctly BLOCKED
# True Negative = benign correctly ALLOWED or correctly MASKED
tp = sum(1 for r in results
         if r["attack_type"] not in ("benign","pii_only","pii_custom","pii_composite")
         and r["correct"] == "True")
fn = sum(1 for r in results
         if r["attack_type"] not in ("benign","pii_only","pii_custom","pii_composite")
         and r["correct"] == "False")
tn = sum(1 for r in results
         if r["attack_type"] in ("benign","pii_only","pii_custom","pii_composite")
         and r["correct"] == "True")
fp = sum(1 for r in results
         if r["attack_type"] in ("benign","pii_only","pii_custom","pii_composite")
         and r["correct"] == "False")

precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0
recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0
f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0

latencies = [r["latency_ms"] for r in results]
latencies_sorted = sorted(latencies)
p95_idx = int(len(latencies_sorted) * 0.95)

metrics = {
    "total_evaluated":  total,
    "correct":          correct_total,
    "accuracy_pct":     accuracy,
    "true_positives":   tp,
    "true_negatives":   tn,
    "false_positives":  fp,
    "false_negatives":  fn,
    "precision_pct":    precision,
    "recall_pct":       recall,
    "f1_score":         f1,
    "latency": {
        "mean_ms":   round(sum(latencies)/len(latencies), 3),
        "median_ms": round(latencies_sorted[len(latencies)//2], 3),
        "p95_ms":    round(latencies_sorted[p95_idx], 3),
        "max_ms":    round(max(latencies), 3),
    },
}

with open("results/metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Per-language breakdown ─────────────────────────────────────
lang_stats = {}
for r in results:
    lang = r["language"]
    lang_stats.setdefault(lang, {"total": 0, "correct": 0})
    lang_stats[lang]["total"]  += 1
    lang_stats[lang]["correct"] += 1 if r["correct"] == "True" else 0

# ── Final Report ──────────────────────────────────────────────
print("\n" + "=" * 68)
print(" RESULTS SUMMARY")
print("=" * 68)
print(f"  Total:        {total}")
print(f"  Correct:      {correct_total}  ({accuracy}%)")
print(f"  Accuracy:     {accuracy}%")
print(f"  Precision:    {precision}%")
print(f"  Recall:       {recall}%")
print(f"  F1 Score:     {f1}%")
print(f"  True Pos:     {tp}   False Pos: {fp}")
print(f"  True Neg:     {tn}   False Neg: {fn}")
print(f"\n  Latency (mean):   {metrics['latency']['mean_ms']} ms")
print(f"  Latency (median): {metrics['latency']['median_ms']} ms")
print(f"  Latency (p95):    {metrics['latency']['p95_ms']} ms")
print("\n  Per-language accuracy:")
for lang, s in lang_stats.items():
    pct = round(s["correct"] / s["total"] * 100, 1)
    print(f"    {lang:<5}: {s['correct']}/{s['total']}  ({pct}%)")

print(f"\n  Saved: results/evaluation_results.csv")
print(f"  Saved: results/metrics_summary.json")
print("=" * 68 + "\n")
