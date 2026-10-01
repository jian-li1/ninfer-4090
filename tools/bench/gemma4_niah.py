#!/usr/bin/env python3
"""Run deterministic full-model Gemma 4 retrieval near the native context limit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from tokenizers import Tokenizer


NEEDLES = (
    ("KEY_917263", "orange-telescope"),
    ("KEY_204811", "violet-compass"),
    ("KEY_731509", "silver-lantern"),
    ("KEY_448027", "cobalt-orchid"),
    ("KEY_665193", "amber-harbor"),
)
RATIOS = (0.05, 0.25, 0.50, 0.75, 0.95)


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def extend_repeated(target: list[int], source: list[int], stop: int) -> None:
    if not source:
        raise RuntimeError("filler token sequence is empty")
    remaining = stop - len(target)
    if remaining < 0:
        raise RuntimeError("fixed prompt material exceeded requested needle position")
    copies, tail = divmod(remaining, len(source))
    target.extend(source * copies)
    target.extend(source[:tail])


def build_prompt(tokenizer: Tokenizer, prompt_tokens: int) -> tuple[list[int], list[dict[str, object]]]:
    header = (
        "<bos><|turn>user\n"
        "You are reading an immutable synthetic archive. Remember every KEY value exactly. "
        "Ignore unrelated archive records.\n"
    )
    filler = (
        "Archive filler record: routine calibration completed normally; this record contains "
        "no key assignment and no requested answer.\n"
    )
    question = (
        "\nEnd of archive. Return the values for KEY_917263, KEY_204811, KEY_731509, "
        "KEY_448027, and KEY_665193 in that order. Reply with only the five values separated "
        "by single spaces.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
    ids = encode(tokenizer, header)
    filler_ids = encode(tokenizer, filler)
    suffix_ids = encode(tokenizer, question)
    positions: list[dict[str, object]] = []

    for ratio, (key, value) in zip(RATIOS, NEEDLES, strict=True):
        desired = round(prompt_tokens * ratio)
        extend_repeated(ids, filler_ids, desired)
        positions.append({"key": key, "value": value, "token": len(ids), "ratio": ratio})
        ids.extend(encode(tokenizer, f"\nImmutable archive fact: {key} = {value}.\n"))

    extend_repeated(ids, filler_ids, prompt_tokens - len(suffix_ids))
    ids.extend(suffix_ids)
    if len(ids) != prompt_tokens:
        raise RuntimeError(f"constructed {len(ids)} tokens, expected {prompt_tokens}")
    return ids, positions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.splitlines()[0].strip())
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def parse_metrics(stdout: str) -> tuple[dict[str, str], list[int]]:
    metric_line = next(
        (line for line in stdout.splitlines() if line.startswith("GEMMA4_LONG_CONTEXT ")), None
    )
    generated_line = next(
        (line for line in stdout.splitlines() if line.startswith("GEMMA4_GENERATED_IDS=")), None
    )
    if metric_line is None or generated_line is None:
        raise RuntimeError("long-context executable did not emit the expected records")
    metrics = dict(field.split("=", 1) for field in metric_line.split()[1:])
    generated = [int(value) for value in generated_line.split("=", 1)[1].split(",")]
    return metrics, generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=240_000)
    parser.add_argument("--generate", type=int, default=96)
    parser.add_argument("--tokens-out", type=Path, default=Path("/tmp/gemma4_niah_tokens.txt"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not 4_096 <= args.prompt_tokens <= 261_120:
        parser.error("--prompt-tokens must leave room below the 262144-token ceiling")
    if not 1 <= args.generate <= 1_024:
        parser.error("--generate must be in [1, 1024]")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt, positions = build_prompt(tokenizer, args.prompt_tokens)
    args.tokens_out.write_text("".join(f"{token}\n" for token in prompt), encoding="ascii")

    peak_used = gpu_used_mib()
    stop_monitor = threading.Event()

    def monitor() -> None:
        nonlocal peak_used
        while not stop_monitor.wait(1.0):
            used = gpu_used_mib()
            if used is not None:
                peak_used = used if peak_used is None else max(peak_used, used)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [
                str(args.executable),
                str(args.artifact),
                "--max-context",
                "262144",
                "--tokens-file",
                str(args.tokens_out),
                "--chunk",
                "64",
                "--generate",
                str(args.generate),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        stop_monitor.set()
        monitor_thread.join()
    elapsed = time.monotonic() - started
    print(completed.stdout, end="")
    if completed.returncode != 0:
        return completed.returncode

    metrics, generated = parse_metrics(completed.stdout)
    generated_text = tokenizer.decode(generated, skip_special_tokens=False)
    normalized = generated_text.casefold()
    missing = [value for _, value in NEEDLES if value.casefold() not in normalized]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = "unknown"
    report = {
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "cache_mode": "local/global-rk4v4-e8",
        "commit": commit,
        "elapsed_seconds": elapsed,
        "generated_text": generated_text,
        "generated_token_ids": generated,
        "metrics": metrics,
        "missing_values": missing,
        "needle_positions": positions,
        "peak_gpu_used_mib": peak_used,
        "prompt_tokens": len(prompt),
        "success": not missing,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        args.report.write_text(rendered, encoding="utf-8")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
