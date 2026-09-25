#!/usr/bin/env python3
"""Post factory usage report for Claude review workflow jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

JOBS = ("review", "fix", "merge")
NA = "n/a"
API_COST_SUFFIX = " (API-equivalent; subscription — not billed per token)"
TABLE_HEADER = (
    "| Job | Turns | Input | Output | Cache read | Cache creation | "
    "API-equiv cost | Runner min (billed) | Linux cost |"
)
TABLE_SEPARATOR = (
    "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"
)


def runner_minutes_elapsed(elapsed_seconds: int) -> int:
    return (elapsed_seconds + 59) // 60


def private_linux_cost_dollars(runner_min: int, rate_per_min: float) -> str:
    return f"{runner_min * rate_per_min:.4f}"


def format_linux_cell(repo_private: bool, runner_min: int, rate_per_min: float) -> str:
    private_linux = private_linux_cost_dollars(runner_min, rate_per_min)
    if repo_private:
        return f"${private_linux}"
    return f"billed: $0.00 (public repo); would be ${private_linux} if private"


def _pick(result: dict[str, Any], path: list[str], label: str, log: Callable[[str], None]) -> str:
    cur: Any = result
    for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            keys = ", ".join(sorted(result.keys())) if isinstance(result, dict) else ""
            log(f"::notice::Missing {label}; result keys: {keys}")
            return NA
        cur = cur[key]
    if cur is None:
        log(f"::notice::Missing {label}; result keys: {', '.join(sorted(result.keys()))}")
        return NA
    return str(cur)


def parse_execution_file(
    exec_file: str | None,
    log: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Return usage fields from Claude execution_file (or n/a defaults)."""
    emit = log or (lambda _msg: None)
    fields = {
        "turns": NA,
        "input": NA,
        "output": NA,
        "cache_read": NA,
        "cache_create": NA,
        "api_cost": NA,
        "duration_ms": NA,
    }
    if not exec_file:
        return fields
    if not os.path.isfile(exec_file):
        emit(f"::notice::Claude execution file path set but file missing: {exec_file}")
        return fields

    try:
        with open(exec_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return fields

    if not isinstance(data, list):
        return fields

    results = [entry for entry in data if isinstance(entry, dict) and entry.get("type") == "result"]
    if not results:
        last_keys = "empty"
        if data and isinstance(data[-1], dict):
            last_keys = ", ".join(sorted(data[-1].keys()))
        emit(f"::notice::No result message in execution file; last entry keys: {last_keys}")
        return fields

    result = results[-1]
    fields["turns"] = _pick(result, ["num_turns"], "num_turns", emit)
    fields["duration_ms"] = _pick(result, ["duration_ms"], "duration_ms", emit)
    fields["input"] = _pick(result, ["usage", "input_tokens"], "usage.input_tokens", emit)
    fields["output"] = _pick(result, ["usage", "output_tokens"], "usage.output_tokens", emit)
    fields["cache_read"] = _pick(
        result, ["usage", "cache_read_input_tokens"], "usage.cache_read_input_tokens", emit
    )
    fields["cache_create"] = _pick(
        result, ["usage", "cache_creation_input_tokens"], "usage.cache_creation_input_tokens", emit
    )
    raw_cost = _pick(result, ["total_cost_usd"], "total_cost_usd", emit)
    if raw_cost != NA:
        fields["api_cost"] = f"~${raw_cost}{API_COST_SUFFIX}"
    return fields


def row_payload(
    job: str,
    usage: dict[str, str],
    runner_min: int,
    linux_cell: str,
) -> dict[str, str]:
    return {
        "job": job,
        "turns": usage["turns"],
        "input": usage["input"],
        "output": usage["output"],
        "cache_read": usage["cache_read"],
        "cache_create": usage["cache_create"],
        "api_cost": usage["api_cost"],
        "runner_min": str(runner_min),
        "linux": linux_cell,
    }


def row_payload_to_json(row: dict[str, str]) -> str:
    return json.dumps(row, separators=(",", ":"))


def format_table_row(row_json: str) -> str:
    row = json.loads(row_json)
    return (
        f"| {row['job']} | {row['turns']} | {row['input']} | {row['output']} | "
        f"{row['cache_read']} | {row['cache_create']} | {row['api_cost']} | "
        f"{row['runner_min']} | {row['linux']} |"
    )


def placeholder_table_row(job: str) -> str:
    return f"| {job} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} |"


def extract_row_json(body: str, job: str) -> str:
    open_tag = f"<!-- factory:usage-data:{job} -->"
    close_tag = f"<!-- /factory:usage-data:{job} -->"
    start = body.find(open_tag)
    if start < 0:
        return ""
    start += len(open_tag)
    end = body.find(close_tag, start)
    if end < 0:
        return ""
    return body[start:end]


def merge_job_rows(existing_body: str, job: str, current_json: str) -> dict[str, str]:
    rows: dict[str, str] = {name: "" for name in JOBS}
    if existing_body:
        for name in JOBS:
            chunk = extract_row_json(existing_body, name)
            if chunk:
                rows[name] = chunk
    rows[job] = current_json
    return rows


def build_table(rows: dict[str, str]) -> str:
    lines = [TABLE_HEADER, f"          {TABLE_SEPARATOR.lstrip()}"]
    for job in JOBS:
        if rows[job]:
            lines.append(format_table_row(rows[job]))
        else:
            lines.append(placeholder_table_row(job))
    return "\n".join(lines)


def build_data_tags(rows: dict[str, str]) -> str:
    parts: list[str] = []
    for job in JOBS:
        if rows[job]:
            parts.append(
                f"<!-- factory:usage-data:{job} -->{rows[job]}<!-- /factory:usage-data:{job} -->"
            )
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


def build_comment_body(pr_number: str | int, rows: dict[str, str]) -> str:
    marker = f"<!-- factory:usage:{pr_number} -->"
    table = build_table(rows)
    data_tags = build_data_tags(rows)
    return (
        f"{marker}\n"
        f"          ### Factory usage (Claude review workflow)\n"
        f"          \n"
        f"          Token **API-equivalent cost** is an estimate only: these jobs use a Claude "
        f"subscription (`CLAUDE_CODE_OAUTH_TOKEN`), not metered API billing.\n"
        f"          \n"
        f"          {table}\n"
        f"          {data_tags}"
    )


def build_step_summary(job: str, duration_ms: str, table: str) -> str:
    return (
        f"### Factory usage — {job}\n\n"
        f"Claude duration (ms): {duration_ms}\n\n"
        f"{table}\n"
    )


def parse_repo_private(value: str | None) -> bool:
    return str(value).lower() == "true"


def _gh_api_json(args: list[str]) -> Any:
    proc = subprocess.run(
        ["gh", "api", *args, "--paginate"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def find_usage_comment(
    repository: str,
    pr_number: str,
    marker: str,
) -> tuple[str | None, str | None]:
    comments = _gh_api_json([f"repos/{repository}/issues/{pr_number}/comments"])
    if not isinstance(comments, list):
        return None, None
    for comment in comments:
        body = comment.get("body") or ""
        if marker in body:
            return str(comment.get("id")), body
    return None, None


def post_or_update_comment(
    repository: str,
    pr_number: str,
    comment_id: str | None,
    body: str,
) -> None:
    payload = json.dumps({"body": body})
    if comment_id and comment_id != "null":
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{repository}/issues/comments/{comment_id}",
                "--input",
                "-",
            ],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
    else:
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{repository}/issues/{pr_number}/comments",
                "--input",
                "-",
            ],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )


def run_report(job: str, emit: Callable[[str], None] | None = None) -> int:
    log = emit or (lambda msg: print(msg, flush=True))

    pr_number = os.environ.get("PR_NUMBER")
    if not pr_number:
        log("PR_NUMBER is required")
        return 1

    rate = float(os.environ.get("ACTIONS_LINUX_RATE_PER_MIN", "0.006"))
    end = int(time.time())
    start = int(os.environ.get("JOB_START", str(end)))
    runner_min = runner_minutes_elapsed(end - start)
    repo_private = parse_repo_private(os.environ.get("REPO_PRIVATE"))
    linux = format_linux_cell(repo_private, runner_min, rate)

    exec_file = os.environ.get("CLAUDE_EXECUTION_FILE") or None
    if exec_file == "":
        exec_file = None
    usage = parse_execution_file(exec_file, log=log)

    current_json = row_payload_to_json(row_payload(job, usage, runner_min, linux))
    marker = f"<!-- factory:usage:{pr_number} -->"
    repository = os.environ["GITHUB_REPOSITORY"]

    comment_id, existing_body = find_usage_comment(repository, pr_number, marker)
    rows = merge_job_rows(existing_body or "", job, current_json)
    table = build_table(rows)
    body = build_comment_body(pr_number, rows)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(build_step_summary(job, usage["duration_ms"], table))

    post_or_update_comment(repository, pr_number, comment_id, body)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post factory Claude review usage report.")
    parser.add_argument(
        "--job",
        required=True,
        choices=JOBS,
        help="Workflow job name (review, fix, or merge)",
    )
    args = parser.parse_args(argv)
    try:
        return run_report(args.job)
    except KeyError as exc:
        print(f"Missing environment variable: {exc.args[0]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
