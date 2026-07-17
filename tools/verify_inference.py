#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def integer(value: str, name: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{name} must be an integer: {value!r}")
    if parsed < minimum:
        fail(f"{name} must be at least {minimum}: {parsed}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one FastWAM LIBERO inference result."
    )
    parser.add_argument("output", type=Path, metavar="OUTPUT")
    parser.add_argument("suite", metavar="SUITE")
    parser.add_argument("task_id", metavar="TASK_ID")
    parser.add_argument("trials", metavar="TRIALS")
    args = parser.parse_args()

    if not args.suite:
        fail("SUITE must not be empty")
    task_id = integer(args.task_id, "TASK_ID", minimum=0)
    trials = integer(args.trials, "TRIALS", minimum=1)
    suite_dir = args.output / args.suite
    if not suite_dir.is_dir():
        fail(f"suite output directory is missing: {suite_dir}")

    candidates = sorted(suite_dir.glob(f"gpu*_task{task_id}_results.json"))
    expected = suite_dir / f"gpu0_task{task_id}_results.json"
    if len(candidates) != 1:
        fail(
            f"expected exactly one result for suite={args.suite!r} task_id={task_id}, "
            f"found {len(candidates)}"
        )
    if candidates[0] != expected:
        fail(f"expected result path {expected}, found {candidates[0]}")

    try:
        with expected.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse result JSON {expected}: {exc}")
    if not isinstance(payload, dict):
        fail("result JSON must contain an object")

    required = ("task_suite", "task_id", "total_episodes", "successes")
    missing = [key for key in required if key not in payload]
    if missing:
        fail(f"result JSON is missing required fields: {', '.join(missing)}")
    for key in ("task_id", "total_episodes", "successes"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            fail(f"result field {key!r} must be an integer")

    if payload["task_suite"] != args.suite:
        fail(
            f"task_suite mismatch: expected {args.suite!r}, "
            f"found {payload['task_suite']!r}"
        )
    if payload["task_id"] != task_id:
        fail(f"task_id mismatch: expected {task_id}, found {payload['task_id']}")
    if payload["total_episodes"] != trials:
        fail(
            f"total_episodes mismatch: expected {trials}, "
            f"found {payload['total_episodes']}"
        )
    successes = payload["successes"]
    if not 0 <= successes <= trials:
        fail(f"successes must be in [0, {trials}], found {successes}")

    print(
        json.dumps(
            {
                "result": str(expected),
                "task_suite": args.suite,
                "task_id": task_id,
                "trials": trials,
                "successes": successes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
