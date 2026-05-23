#!/usr/bin/env python3
"""Extract per-step + warm aggregate metrics from a TorchSpec Modal run's
terminal log (the stdout stream from `modal run --detach ...`).

Usage:
    python extract_modal_perf.py /path/to/terminal_log.txt --label D1
    python extract_modal_perf.py log1.txt log2.txt --json out.json
    python extract_modal_perf.py log1.txt log2.txt --markdown

The script never talks to WandB; it parses the tqdm lines + structured log
messages already present in the local stream:

    Training:  11%|...| 555/5000 [05:03<1:06:38, 1.18step/s, loss=..., acc=..., thru=..., I=..., T=..., wait=..., pool=...]
    COMPUTE_BREAKDOWN step=N: forward=Xms backward=Yms
    [start] Starting: num_steps=5000, ..., global_batch_size=8, ...
    [_run_training/_train_impl printouts]
    [exit summary in the modal terminal footer: exit_code: 0]

Output:
    A single JSON record per log (or a Markdown table for human reading)
    with these fields:
        run_label, total_steps, final_step, completed (bool),
        warm_step_time_s, warm_throughput_samples_per_s,
        median_loss, median_acc, final_loss_mean, final_acc_mean,
        median_compute_time_s, median_compute_fwd_ms, median_compute_bwd_ms,
        median_data_pool_wait_s,
        median_infer_capacity, median_train_capacity, median_I_over_T,
        median_pool, min_pool, max_pool,
        nan_events, oom_events, runtime_errors, exit_code,
        global_batch_size, num_steps_target, dp_size,
        eta_remaining_seconds (last value, if not yet done)

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# Example tqdm line (after ANSI strip):
# Training:  11%|...| 555/5000 [05:03<1:06:38,  1.18step/s, loss=5.682, acc=0.102, acc_len=0.00, thru=10.0, I=61.0, T=9.6, wait=0.0s, pool=64, epoch=1/1]
# Eagle3 variant:
# Training:   0%|...| 10/5000 [02:30<5:21:42,  3.87s/step, loss=11.413, acc=0.040, ...]
_TQDM_RE = re.compile(
    r"Training:\s*\d+%\|[^|]*\|\s*"
    r"(?P<step>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*"
    r"(?P<rate>[\d.]+)\s*(?P<rate_unit>step/s|s/step)"
    r"(?P<rest>[^\]]*)\]"
)

_FLOAT = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"

_FIELD_REGEXES = {
    "loss": re.compile(rf"loss=({_FLOAT})"),
    "acc": re.compile(rf"acc=({_FLOAT})"),
    "acc_len": re.compile(rf"acc_len=({_FLOAT})"),
    "thru": re.compile(rf"thru=({_FLOAT})"),
    "I": re.compile(rf"I=({_FLOAT})"),
    "T": re.compile(rf"T=({_FLOAT})"),
    "wait": re.compile(rf"wait=({_FLOAT})s"),
    "pool": re.compile(r"pool=(\d+)"),
}

_COMPUTE_RE = re.compile(
    r"COMPUTE_BREAKDOWN step=(?P<step>\d+):\s*forward=(?P<fwd>[\d.]+)ms\s*backward=(?P<bwd>[\d.]+)ms"
)

# Authoritative per-step record (logged by loop.py):
# TIMING step=5000: step=0.925s data=0.482s compute=0.840s [fwd=0.376s bwd=0.438s opt=0.024s] dispatch=0.071s
_TIMING_RE = re.compile(
    r"TIMING step=(?P<step>\d+):\s*"
    r"step=(?P<step_s>[\d.]+)s\s*"
    r"data=(?P<data_s>[\d.]+)s\s*"
    r"compute=(?P<compute_s>[\d.]+)s\s*"
    r"\[fwd=(?P<fwd_s>[\d.]+)s\s*"
    r"bwd=(?P<bwd_s>[\d.]+)s\s*"
    r"opt=(?P<opt_s>[\d.]+)s\]\s*"
    r"dispatch=(?P<dispatch_s>[\d.]+)s"
)

_TRAINING_COMPLETE_RE = re.compile(
    r"Training completed:\s*(?P<steps>\d+)\s*steps in\s*(?P<seconds>[\d.]+)s"
    r"(?:.*?avg inference=(?P<avg_infer>[\d.]+)\s*entries/s)?"
    r"(?:.*?avg training=(?P<avg_train>[\d.]+)\s*entries/s)?"
)

_START_RE = re.compile(
    r"Starting: num_steps=(?P<num_steps>\d+),\s*num_epochs=\d+,\s*steps_per_epoch=\d+,"
    r"\s*global_batch_size=(?P<gbs>\d+),\s*accumulation_steps=(?P<accum>\d+),"
    r"\s*dp_size=(?P<dp>\d+),\s*per_dp_rank_batch_size=(?P<pdrb>\d+)"
)

_EXIT_RE = re.compile(r"exit_code:\s*(\d+)")
_ELAPSED_RE = re.compile(r"elapsed_ms:\s*(\d+)")


_NAN_NEEDLES = ("NaN", "nan_loss", "ValueError: nan")
_OOM_NEEDLES = (
    "OutOfMemoryError",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
)
_FATAL_NEEDLES = (
    "RuntimeError",
    "AssertionError",
    "FATAL",
    "Segmentation fault",
    "SIGSEGV",
    "FAILED:",
)


@dataclass
class _Sample:
    step: int
    elapsed_s: float
    eta_s: Optional[float]
    rate_step_per_s: float
    loss: Optional[float] = None
    acc: Optional[float] = None
    acc_len: Optional[float] = None
    thru: Optional[float] = None
    infer_cap: Optional[float] = None
    train_cap: Optional[float] = None
    wait: Optional[float] = None
    pool: Optional[int] = None


@dataclass
class _Compute:
    step: int
    forward_ms: float
    backward_ms: float


@dataclass
class _Timing:
    step: int
    step_s: float
    data_s: float
    compute_s: float
    fwd_s: float
    bwd_s: float
    opt_s: float
    dispatch_s: float


@dataclass
class RunSummary:
    log_path: str
    run_label: str
    num_steps_target: int = 0
    global_batch_size: int = 0
    accumulation_steps: int = 0
    dp_size: int = 0
    per_dp_rank_batch_size: int = 0
    final_step: int = 0
    completed: bool = False
    exit_code: Optional[int] = None
    elapsed_ms: Optional[int] = None
    warm_step_time_s: Optional[float] = None
    warm_throughput_samples_per_s: Optional[float] = None
    median_loss: Optional[float] = None
    median_acc: Optional[float] = None
    median_acc_len: Optional[float] = None
    final_loss_mean: Optional[float] = None
    final_acc_mean: Optional[float] = None
    final_acc_len_mean: Optional[float] = None
    median_compute_fwd_ms: Optional[float] = None
    median_compute_bwd_ms: Optional[float] = None
    median_compute_total_s: Optional[float] = None
    median_dispatch_wait_s: Optional[float] = None
    median_infer_capacity: Optional[float] = None
    median_train_capacity: Optional[float] = None
    median_I_over_T: Optional[float] = None
    median_pool: Optional[float] = None
    min_pool: Optional[int] = None
    max_pool: Optional[int] = None
    nan_events: int = 0
    oom_events: int = 0
    runtime_errors: int = 0
    last_eta_seconds: Optional[float] = None
    warmup_n: int = 100
    n_samples_total: int = 0
    n_samples_warm: int = 0
    n_timing_total: int = 0
    n_timing_warm: int = 0
    error_excerpts: list[str] = field(default_factory=list)
    # From "Training completed: …" line (authoritative wall-clock):
    completed_steps: Optional[int] = None
    completed_seconds: Optional[float] = None
    completed_avg_infer_per_s: Optional[float] = None
    completed_avg_train_per_s: Optional[float] = None
    # From TIMING step=N: lines (authoritative per-step):
    timing_warm_step_s: Optional[float] = None
    timing_warm_data_s: Optional[float] = None
    timing_warm_compute_s: Optional[float] = None
    timing_warm_fwd_s: Optional[float] = None
    timing_warm_bwd_s: Optional[float] = None
    timing_warm_opt_s: Optional[float] = None
    timing_warm_dispatch_s: Optional[float] = None
    timing_warm_throughput_samples_per_s: Optional[float] = None


def _parse_elapsed(s: str) -> float:
    parts = s.split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    return float(parts[0])


def _parse_eta(s: str) -> Optional[float]:
    if "?" in s:
        return None
    try:
        return _parse_elapsed(s)
    except Exception:
        return None


def _maybe_float(rx: re.Pattern[str], text: str) -> Optional[float]:
    m = rx.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _maybe_int(rx: re.Pattern[str], text: str) -> Optional[int]:
    m = rx.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_log(path: str, run_label: str, warmup_n: int = 100) -> RunSummary:
    summary = RunSummary(log_path=path, run_label=run_label, warmup_n=warmup_n)

    samples: list[_Sample] = []
    computes: list[_Compute] = []
    timings: list[_Timing] = []
    error_excerpts: list[str] = []

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = _strip_ansi(raw.rstrip("\n"))

            for needle in _NAN_NEEDLES:
                if needle in line:
                    summary.nan_events += 1
                    if len(error_excerpts) < 5:
                        error_excerpts.append(f"NaN: {line.strip()[:200]}")
                    break
            for needle in _OOM_NEEDLES:
                if needle in line:
                    summary.oom_events += 1
                    if len(error_excerpts) < 5:
                        error_excerpts.append(f"OOM: {line.strip()[:200]}")
                    break
            for needle in _FATAL_NEEDLES:
                if needle in line:
                    summary.runtime_errors += 1
                    if len(error_excerpts) < 5:
                        error_excerpts.append(f"FATAL: {line.strip()[:200]}")
                    break

            m = _START_RE.search(line)
            if m:
                summary.num_steps_target = int(m.group("num_steps"))
                summary.global_batch_size = int(m.group("gbs"))
                summary.accumulation_steps = int(m.group("accum"))
                summary.dp_size = int(m.group("dp"))
                summary.per_dp_rank_batch_size = int(m.group("pdrb"))

            m = _EXIT_RE.search(line)
            if m:
                summary.exit_code = int(m.group(1))
                summary.completed = summary.exit_code == 0
            m = _ELAPSED_RE.search(line)
            if m:
                summary.elapsed_ms = int(m.group(1))

            m = _COMPUTE_RE.search(line)
            if m:
                computes.append(
                    _Compute(
                        step=int(m.group("step")),
                        forward_ms=float(m.group("fwd")),
                        backward_ms=float(m.group("bwd")),
                    )
                )

            m = _TIMING_RE.search(line)
            if m:
                timings.append(
                    _Timing(
                        step=int(m.group("step")),
                        step_s=float(m.group("step_s")),
                        data_s=float(m.group("data_s")),
                        compute_s=float(m.group("compute_s")),
                        fwd_s=float(m.group("fwd_s")),
                        bwd_s=float(m.group("bwd_s")),
                        opt_s=float(m.group("opt_s")),
                        dispatch_s=float(m.group("dispatch_s")),
                    )
                )

            m = _TRAINING_COMPLETE_RE.search(line)
            if m:
                summary.completed_steps = int(m.group("steps"))
                summary.completed_seconds = float(m.group("seconds"))
                if m.group("avg_infer"):
                    summary.completed_avg_infer_per_s = float(m.group("avg_infer"))
                if m.group("avg_train"):
                    summary.completed_avg_train_per_s = float(m.group("avg_train"))
                summary.completed = True

            m = _TQDM_RE.search(line)
            if m:
                try:
                    elapsed_s = _parse_elapsed(m.group("elapsed"))
                except Exception:
                    elapsed_s = 0.0
                eta_s = _parse_eta(m.group("eta"))
                rate = float(m.group("rate"))
                if m.group("rate_unit") == "s/step":
                    rate_step_per_s = 1.0 / rate if rate > 0 else 0.0
                else:
                    rate_step_per_s = rate
                rest = m.group("rest")
                sample = _Sample(
                    step=int(m.group("step")),
                    elapsed_s=elapsed_s,
                    eta_s=eta_s,
                    rate_step_per_s=rate_step_per_s,
                    loss=_maybe_float(_FIELD_REGEXES["loss"], rest),
                    acc=_maybe_float(_FIELD_REGEXES["acc"], rest),
                    acc_len=_maybe_float(_FIELD_REGEXES["acc_len"], rest),
                    thru=_maybe_float(_FIELD_REGEXES["thru"], rest),
                    infer_cap=_maybe_float(_FIELD_REGEXES["I"], rest),
                    train_cap=_maybe_float(_FIELD_REGEXES["T"], rest),
                    wait=_maybe_float(_FIELD_REGEXES["wait"], rest),
                    pool=_maybe_int(_FIELD_REGEXES["pool"], rest),
                )
                samples.append(sample)

    summary.error_excerpts = error_excerpts
    summary.n_samples_total = len(samples)

    if samples:
        summary.final_step = samples[-1].step
        summary.last_eta_seconds = samples[-1].eta_s

    # Warm window: step >= warmup_n
    warm = [s for s in samples if s.step >= warmup_n]
    summary.n_samples_warm = len(warm)

    if warm:
        rates = [s.rate_step_per_s for s in warm if s.rate_step_per_s > 0]
        if rates:
            warm_step_per_s = statistics.median(rates)
            if warm_step_per_s > 0:
                summary.warm_step_time_s = 1.0 / warm_step_per_s
                if summary.global_batch_size:
                    summary.warm_throughput_samples_per_s = (
                        summary.global_batch_size * warm_step_per_s
                    )

        losses = [s.loss for s in warm if s.loss is not None and not math.isnan(s.loss)]
        accs = [s.acc for s in warm if s.acc is not None and not math.isnan(s.acc)]
        acc_lens = [s.acc_len for s in warm if s.acc_len is not None and not math.isnan(s.acc_len)]
        thrus = [s.thru for s in warm if s.thru is not None]
        Is = [s.infer_cap for s in warm if s.infer_cap is not None and s.infer_cap > 0]
        Ts = [s.train_cap for s in warm if s.train_cap is not None and s.train_cap > 0]
        waits = [s.wait for s in warm if s.wait is not None]
        pools = [s.pool for s in warm if s.pool is not None]

        if losses:
            summary.median_loss = statistics.median(losses)
        if accs:
            summary.median_acc = statistics.median(accs)
        if acc_lens:
            summary.median_acc_len = statistics.median(acc_lens)

        last_100 = warm[-100:]
        last_losses = [s.loss for s in last_100 if s.loss is not None and not math.isnan(s.loss)]
        last_accs = [s.acc for s in last_100 if s.acc is not None and not math.isnan(s.acc)]
        last_acc_lens = [s.acc_len for s in last_100 if s.acc_len is not None]
        if last_losses:
            summary.final_loss_mean = sum(last_losses) / len(last_losses)
        if last_accs:
            summary.final_acc_mean = sum(last_accs) / len(last_accs)
        if last_acc_lens:
            summary.final_acc_len_mean = sum(last_acc_lens) / len(last_acc_lens)

        if waits:
            summary.median_dispatch_wait_s = statistics.median(waits)
        if Is:
            summary.median_infer_capacity = statistics.median(Is)
        if Ts:
            summary.median_train_capacity = statistics.median(Ts)
        if Is and Ts and summary.median_train_capacity:
            summary.median_I_over_T = (
                summary.median_infer_capacity / summary.median_train_capacity
            )
        if pools:
            summary.median_pool = statistics.median(pools)
            summary.min_pool = min(pools)
            summary.max_pool = max(pools)

    warm_computes = [c for c in computes if c.step >= warmup_n]
    if warm_computes:
        fwds = [c.forward_ms for c in warm_computes]
        bwds = [c.backward_ms for c in warm_computes]
        summary.median_compute_fwd_ms = statistics.median(fwds)
        summary.median_compute_bwd_ms = statistics.median(bwds)
        summary.median_compute_total_s = (
            (summary.median_compute_fwd_ms + summary.median_compute_bwd_ms) / 1000.0
        )

    summary.n_timing_total = len(timings)
    warm_timings = [t for t in timings if t.step >= warmup_n]
    summary.n_timing_warm = len(warm_timings)
    if warm_timings:
        summary.timing_warm_step_s = statistics.median(t.step_s for t in warm_timings)
        summary.timing_warm_data_s = statistics.median(t.data_s for t in warm_timings)
        summary.timing_warm_compute_s = statistics.median(t.compute_s for t in warm_timings)
        summary.timing_warm_fwd_s = statistics.median(t.fwd_s for t in warm_timings)
        summary.timing_warm_bwd_s = statistics.median(t.bwd_s for t in warm_timings)
        summary.timing_warm_opt_s = statistics.median(t.opt_s for t in warm_timings)
        summary.timing_warm_dispatch_s = statistics.median(t.dispatch_s for t in warm_timings)
        if summary.global_batch_size and summary.timing_warm_step_s:
            summary.timing_warm_throughput_samples_per_s = (
                summary.global_batch_size / summary.timing_warm_step_s
            )

    return summary


def to_markdown_table(summaries: list[RunSummary]) -> str:
    def fmt(v, prec=2):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            return f"{v:.{prec}f}"
        return str(v)

    rows = []
    rows.append(
        "| Run | done | final step / target | wall (s) | step (s) | thru (samp/s) | fwd (s) | bwd (s) | opt (s) | data (s) | dispatch (s) | I cap | T cap | I/T | pool med / min–max | loss @ end | acc @ end | acc_len @ end | NaN | OOM | exit |"
    )
    rows.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"
    )
    for s in summaries:
        pool_cell = "—"
        if s.median_pool is not None:
            pool_cell = (
                f"{int(round(s.median_pool))} / {s.min_pool}–{s.max_pool}"
            )
        # Prefer TIMING-derived numbers (authoritative) over tqdm-derived ones.
        step_s = s.timing_warm_step_s or s.warm_step_time_s
        thru = s.timing_warm_throughput_samples_per_s or s.warm_throughput_samples_per_s
        rows.append(
            "| " + " | ".join([
                s.run_label,
                fmt(s.completed),
                f"{s.completed_steps or s.final_step} / {s.num_steps_target}",
                fmt(s.completed_seconds, 1),
                fmt(step_s, 3),
                fmt(thru, 2),
                fmt(s.timing_warm_fwd_s, 3),
                fmt(s.timing_warm_bwd_s, 3),
                fmt(s.timing_warm_opt_s, 3),
                fmt(s.timing_warm_data_s, 3),
                fmt(s.timing_warm_dispatch_s, 3),
                fmt(s.median_infer_capacity, 1),
                fmt(s.median_train_capacity, 2),
                fmt(s.median_I_over_T, 2),
                pool_cell,
                fmt(s.final_loss_mean, 3),
                fmt(s.final_acc_mean, 4),
                fmt(s.final_acc_len_mean, 3),
                str(s.nan_events),
                str(s.oom_events),
                "—" if s.exit_code is None else str(s.exit_code),
            ]) + " |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="One or more terminal log files")
    parser.add_argument(
        "--label", action="append", default=None,
        help="One label per log (in matching order). If omitted, the file basename is used.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--json", type=str, default=None, help="Path to write a JSON array.")
    parser.add_argument("--markdown", action="store_true", help="Print a markdown summary table.")
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.logs):
        print("ERROR: --label count must equal log count", file=sys.stderr)
        return 2

    labels = args.label or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]

    summaries = [parse_log(p, lbl, warmup_n=args.warmup) for p, lbl in zip(args.logs, labels)]

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        print(f"wrote {args.json}")

    if args.markdown or not args.json:
        print(to_markdown_table(summaries))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
