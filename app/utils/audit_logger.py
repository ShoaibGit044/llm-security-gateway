"""
app/utils/audit_logger.py
Structured JSON audit logging for every gateway request.

Every request is logged as one JSON line (JSONL format) to
results/audit_log.jsonl. This makes the system auditable —
the instructor can inspect every decision the gateway made.

Log format matches the assignment's required JSON response spec:
  input_id, language, rule_score, semantic_score, pii_entities,
  final_risk, decision, safe_text, reason_codes, latency_ms
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Writes one JSON record per request to a .jsonl file.
    JSONL = one JSON object per line, easy to process with pandas.
    """

    def __init__(self, log_file: str = "results/audit_log.jsonl"):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        """
        Append one audit record to the log file.

        record fields:
          input_id       -- unique request identifier
          timestamp      -- ISO 8601 UTC timestamp
          language       -- detected language code (en, ur, ko...)
          original_text  -- raw user input (first 200 chars)
          rule_score     -- float from rule-based detector
          semantic_score -- float from ML detector
          final_risk     -- combined risk score
          pii_entities   -- list of detected PII entity types
          decision       -- ALLOWED | MASKED | BLOCKED
          safe_text      -- output after masking (or null)
          reason_codes   -- list of reason strings
          latency_ms     -- total processing time in ms
        """
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Audit log write failed: %s", e)

    def read_all(self) -> list:
        """Read all audit records back as a list of dicts."""
        records = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records


# Module-level singleton
_audit_logger = None


def get_audit_logger(log_file: str = "results/audit_log.jsonl") -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file)
    return _audit_logger
