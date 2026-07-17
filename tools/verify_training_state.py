#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn


LOSS_RE = re.compile(
    r"\[train\][ \t]+epoch=\d+[ \t]+step=(\d+)/(\d+)[ \t]+"
    r"loss=([^ \t\r\n]+)"
)
RESUME_EVIDENCE = (
    "Resuming full training state from directory",
    "Restored dataloader progress",
    "Loaded accelerate training state",
)


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"STEP must be an integer: {value!r}")
    if parsed < 0:
        fail(f"STEP must be nonnegative: {parsed}")
    return parsed


def require_nonempty_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"required regular file is missing: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        fail(f"cannot stat required file {path}: {exc}")
    if size == 0:
        fail(f"required file is empty: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a FastWAM accelerate training-state checkpoint."
    )
    parser.add_argument("state_dir", type=Path, metavar="STATE_DIR")
    parser.add_argument("step", metavar="STEP")
    parser.add_argument("log", type=Path, metavar="LOG")
    parser.add_argument("mode", choices=("fresh", "resumed"))
    args = parser.parse_args()

    step = nonnegative_integer(args.step)
    if not args.state_dir.is_dir():
        fail(f"training state directory is missing: {args.state_dir}")

    trainer_state = args.state_dir / "trainer_state.json"
    required = (
        trainer_state,
        args.state_dir / "scheduler.bin",
        args.state_dir / "random_states_0.pkl",
        args.state_dir / "pytorch_model" / "mp_rank_00_model_states.pt",
    )
    for path in required:
        require_nonempty_regular_file(path)

    optimizer_states = sorted(
        (args.state_dir / "pytorch_model").glob("*optim_states.pt")
    )
    if not optimizer_states:
        fail(
            "no optimizer state matching pytorch_model/*optim_states.pt "
            f"under {args.state_dir}"
        )
    for path in optimizer_states:
        require_nonempty_regular_file(path)

    try:
        with trainer_state.open("r", encoding="utf-8") as handle:
            state_payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse trainer state JSON {trainer_state}: {exc}")
    if not isinstance(state_payload, dict):
        fail("trainer_state.json must contain an object")
    global_step = state_payload.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        fail("trainer_state.json global_step must be an integer")
    if global_step != step:
        fail(f"global_step mismatch: expected {step}, found {global_step}")

    require_nonempty_regular_file(args.log)
    try:
        log_text = args.log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(f"cannot read training log {args.log}: {exc}")

    matching_losses = []
    for match in LOSS_RE.finditer(log_text):
        if int(match.group(1)) != step:
            continue
        try:
            loss = float(match.group(3))
        except ValueError:
            fail(f"invalid loss at step {step}: {match.group(3)!r}")
        if not math.isfinite(loss):
            fail(f"loss at step {step} is not finite: {match.group(3)!r}")
        matching_losses.append(loss)
    if not matching_losses:
        fail(f"no finite training loss found for expected step {step}")

    if args.mode == "resumed":
        missing_evidence = [text for text in RESUME_EVIDENCE if text not in log_text]
        if missing_evidence:
            fail(
                "resumed log is missing recovery evidence: "
                + ", ".join(missing_evidence)
            )

    print(
        json.dumps(
            {
                "global_step": global_step,
                "loss": matching_losses[-1],
                "mode": args.mode,
                "optimizer_state_files": len(optimizer_states),
                "state_dir": str(args.state_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
