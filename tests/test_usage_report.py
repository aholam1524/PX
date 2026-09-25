"""Tests for .github/scripts/usage_report.py pure helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

import usage_report  # noqa: E402


def test_parse_execution_file_normal(tmp_path):
    exec_path = tmp_path / "exec.json"
    exec_path.write_text(
        json.dumps(
            [
                {"type": "assistant", "message": "hi"},
                {
                    "type": "result",
                    "num_turns": 3,
                    "duration_ms": 12000,
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                    },
                    "total_cost_usd": 0.42,
                },
            ]
        ),
        encoding="utf-8",
    )
    usage = usage_report.parse_execution_file(str(exec_path))
    assert usage["turns"] == "3"
    assert usage["duration_ms"] == "12000"
    assert usage["input"] == "100"
    assert usage["output"] == "50"
    assert usage["cache_read"] == "10"
    assert usage["cache_create"] == "5"
    assert usage["api_cost"] == "~$0.42 (API-equivalent; subscription — not billed per token)"


def test_parse_execution_file_missing_path():
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    usage = usage_report.parse_execution_file("/no/such/file.json", log=log)
    assert usage["turns"] == usage_report.NA
    assert any("file missing" in line for line in logs)


def test_parse_execution_file_missing_fields(tmp_path):
    exec_path = tmp_path / "exec.json"
    exec_path.write_text(json.dumps([{"type": "result", "usage": {}}]), encoding="utf-8")
    logs: list[str] = []
    usage = usage_report.parse_execution_file(str(exec_path), log=logs.append)
    assert usage["turns"] == usage_report.NA
    assert usage["input"] == usage_report.NA
    assert usage["api_cost"] == usage_report.NA
    assert any("Missing num_turns" in line for line in logs)


@pytest.mark.parametrize(
    ("elapsed", "expected_min"),
    [
        (0, 0),
        (1, 1),
        (59, 1),
        (60, 1),
        (61, 2),
    ],
)
def test_runner_minutes_round_up(elapsed, expected_min):
    assert usage_report.runner_minutes_elapsed(elapsed) == expected_min


def test_linux_cell_private():
    cell = usage_report.format_linux_cell(True, 10, 0.006)
    assert cell == "$0.0600"


def test_linux_cell_public():
    cell = usage_report.format_linux_cell(False, 10, 0.006)
    assert cell == "billed: $0.00 (public repo); would be $0.0600 if private"


def test_merge_second_job_into_existing_comment():
    review_json = usage_report.row_payload_to_json(
        usage_report.row_payload(
            "review",
            {
                "turns": "2",
                "input": "1",
                "output": "2",
                "cache_read": "3",
                "cache_create": "4",
                "api_cost": "n/a",
                "duration_ms": "100",
            },
            runner_min=5,
            linux_cell="$0.0300",
        )
    )
    existing = usage_report.build_comment_body("42", {"review": review_json, "fix": "", "merge": ""})

    fix_json = usage_report.row_payload_to_json(
        usage_report.row_payload(
            "fix",
            {
                "turns": "1",
                "input": "10",
                "output": "20",
                "cache_read": "0",
                "cache_create": "0",
                "api_cost": "n/a",
                "duration_ms": "200",
            },
            runner_min=2,
            linux_cell="billed: $0.00 (public repo); would be $0.0120 if private",
        )
    )
    rows = usage_report.merge_job_rows(existing, "fix", fix_json)
    body = usage_report.build_comment_body("42", rows)

    assert "<!-- factory:usage:42 -->" in body
    assert usage_report.extract_row_json(body, "review") == review_json
    assert usage_report.extract_row_json(body, "fix") == fix_json
    assert "| review | 2 | 1 | 2 | 3 | 4 | n/a | 5 | $0.0300 |" in body
    assert "| fix | 1 | 10 | 20 | 0 | 0 | n/a | 2 | billed: $0.00 (public repo); would be $0.0120 if private |" in body
    assert "<!-- factory:usage-data:review -->" in body
    assert "<!-- factory:usage-data:fix -->" in body
