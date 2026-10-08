#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Bench result parser: reads cann-bench JSON reports
and extracts fields for routing decisions.
"""

import csv
import glob
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


def parse_precision_result(path: str) -> Dict[str, Any]:
    """Read precision_result.json."""
    if not os.path.exists(path):
        return {
            "precision_overall": False,
            "_error": f"file missing: {path}",
        }
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_perf_result(path: str) -> Dict[str, Any]:
    """Read perf_result.json."""
    if not os.path.exists(path):
        return {
            "perf_pass": False,
            "_error": f"file missing: {path}",
        }
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check_build_success(path: str) -> bool:
    """Check build.log for STATUS: SUCCESS."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return "STATUS: SUCCESS" in content


def _read_kernel_names(csv_path, keywords):
    """Read kernel names from CSV, split into all kernels and custom-only."""
    all_kernels = []
    custom_only = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            all_kernels.append(name)
            if not any(kw in name for kw in keywords):
                custom_only.append(name)
    return all_kernels, custom_only


def _parse_kernel_csv(case, case_idx, keywords):
    """Parse one kernel_details.csv, return summary lines."""
    csv_path = case.get("kernel_csv", "")
    case_id = case.get("case_id", f"case_{case_idx}")
    if not csv_path or not os.path.exists(csv_path):
        return [f"- {case_id}: kernel_csv missing"]

    lines = []
    try:
        kernels, custom_kernels = _read_kernel_names(csv_path, keywords)

        lines.append(
            f"- {case_id}: {len(kernels)} kernels, "
            f"{len(custom_kernels)} custom"
        )
        if custom_kernels:
            top3 = ", ".join(custom_kernels[:3])
            lines.append(f"  custom: {top3}")
        else:
            prefixes = list(set(
                k.split("_")[0] + "_..."
                for k in kernels[:5]
            ))
            names_str = ", ".join(prefixes[:5])
            lines.append(
                f"  all aclnn: {names_str}"
            )
            lines.append(
                "  -> no @pl.jit kernel, "
                "anti-cheat type A1/A2"
            )
    except (OSError, csv.Error) as exc:
        lines.append(f"- {case_id}: csv parse failed ({exc})")
    return lines


def analyze_kernel_csv_for_anticheat(
    perf_result: Dict[str, Any], max_cases: int = 3,
) -> str:
    """Analyse worst_6_cases kernel CSVs for anti-cheat."""
    w6 = perf_result.get("worst_6_cases", [])
    if not w6:
        return "(no worst_6_cases data)"

    aclnn_keywords = ["aclnn", "CacheClean", "CannBenchCacheClean"]
    lines = []
    for i, case in enumerate(w6[:max_cases]):
        lines.extend(
            _parse_kernel_csv(case, i, aclnn_keywords)
        )

    summary = "\n".join(lines)
    has_custom = any("custom:" in ln for ln in lines)
    if not has_custom:
        summary += (
            "\n\nConclusion: no custom @pl.jit kernel "
            "found. Must rewrite core logic with @pl.jit."
        )
    return summary


# -------------------------------------------------------
# cann-bench execution + report parsing
# -------------------------------------------------------

def _find_good_ctypes_dir() -> str:
    """Find a working _ctypes.so lib-dynload directory."""
    candidates = [
        "/usr/lib/python3.12/lib-dynload",
        "/usr/lib/python3/dist-packages",
    ]
    for d in glob.glob("/usr/lib/python3.*/lib-dynload"):
        if d not in candidates:
            candidates.append(d)
    for d in candidates:
        so = os.path.join(
            d, "_ctypes.cpython-312-x86_64-linux-gnu.so",
        )
        if os.path.exists(so):
            return d
    return ""


_ASCEND_LD_LIBS = [
    "/home/developer/Ascend/cann-9.1.0/lib64",
    "/home/developer/Ascend/cann-9.1.0/x86_64-linux/lib64",
    "/home/developer/Ascend/cann-9.1.0/lib64/plugin/opskernel",
    "/home/developer/Ascend/cann-9.1.0/lib64/plugin/nnengine",
    "/usr/local/Ascend/driver/lib64",
    "/usr/local/Ascend/driver/lib64/common",
    "/usr/local/Ascend/driver/lib64/driver",
]


def _build_eval_env() -> Dict[str, str]:
    """Build env vars for cann-bench evaluation."""
    env = os.environ.copy()
    cann_py = "/usr/local/Ascend/cann/python/site-packages"
    cann91_py = (
        "/home/developer/Ascend/cann-9.1.0"
        "/python/site-packages"
    )
    good_ctypes = _find_good_ctypes_dir()
    pypath_parts = (
        ([good_ctypes] if good_ctypes else [])
        + [cann_py, cann91_py]
    )
    env["PYTHONPATH"] = ":".join(pypath_parts)
    env["LD_LIBRARY_PATH"] = ":".join(_ASCEND_LD_LIBS)
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    env["TILE_FWK_DEVICE_ID"] = device_id
    cann_bin = "/home/developer/Ascend/cann-9.1.0/bin"
    ccec_bin = (
        "/home/developer/Ascend/cann-9.1.0"
        "/tools/ccec_compiler/bin"
    )
    sys_path = os.environ.get(
        "PATH", "/usr/local/bin:/usr/bin:/bin",
    )
    env["PATH"] = f"{cann_bin}:{ccec_bin}:{sys_path}"
    env.pop("LD_PRELOAD", None)
    return env


def _find_latest_report(
    reports_dir: str, op_name: str,
    suffix: str = ".json", after_ts: float = 0,
) -> Optional[str]:
    """Find the latest eval report in reports dir."""
    pattern = os.path.join(
        reports_dir, f"{op_name}_eval_*{suffix}",
    )
    files = sorted(glob.glob(pattern), reverse=True)
    if after_ts > 0:
        files = [
            f for f in files
            if os.path.getmtime(f) > after_ts
        ]
    return files[0] if files else None


def _run_eval_cmd(cmd, env, cwd, log_callback, label):
    """Run an eval subprocess and log results."""
    if log_callback:
        log_callback(f"Running {label}: {' '.join(cmd[:6])}...")

    start_ts = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=cwd, timeout=3600,
    )

    if log_callback:
        log_callback(
            f"{label} done, returncode={proc.returncode}"
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-500:]
            stdout_tail = (proc.stdout or "")[-500:]
            log_callback(f"stderr: {stderr_tail}")
            log_callback(f"stdout: {stdout_tail}")

    return proc, start_ts


def run_perf_eval(
    task_dir: str, cannbench_src: str,
    device_id: int = 0, log_callback=None,
) -> Tuple[bool, str, str]:
    """Run cann-bench perf evaluation directly."""
    cmd = [
        "python3", "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--task-dir", task_dir,
        "--device-id", str(device_id),
        "--warmup", "2", "--repeat", "3",
        "--perf-metric-strategy", "trace_view",
    ]
    env = _build_eval_env()
    env["PYTHONPATH"] = (
        f"{cannbench_src}:{cannbench_src}"
        f"/cann_bench_utils:{env['PYTHONPATH']}"
    )

    proc, start_ts = _run_eval_cmd(
        cmd, env, cannbench_src, log_callback, "perf eval",
    )

    op_name = os.path.basename(task_dir.rstrip("/"))
    reports_dir = os.path.join(cannbench_src, "reports")
    report_path = _find_latest_report(
        reports_dir, op_name, ".json", after_ts=start_ts,
    )

    if proc.returncode != 0 and not report_path:
        if log_callback:
            log_callback(
                f"eval failed (rc={proc.returncode}), "
                "no new report"
            )

    combined = proc.stdout + proc.stderr
    return proc.returncode == 0, combined, report_path or ""


def run_precision_eval(
    task_dir: str, cannbench_src: str,
    device_id: int = 0, log_callback=None,
) -> Tuple[bool, str, str]:
    """Run cann-bench precision evaluation directly."""
    cmd = [
        "python3", "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--task-dir", task_dir,
        "--device-id", str(device_id),
        "--no-perf",
    ]
    env = _build_eval_env()
    env["PYTHONPATH"] = (
        f"{cannbench_src}:{cannbench_src}"
        f"/cann_bench_utils:{env['PYTHONPATH']}"
    )

    proc, start_ts = _run_eval_cmd(
        cmd, env, cannbench_src, log_callback, "precision eval",
    )

    op_name = os.path.basename(task_dir.rstrip("/"))
    reports_dir = os.path.join(cannbench_src, "reports")
    report_path = _find_latest_report(
        reports_dir, op_name, ".json", after_ts=start_ts,
    )

    combined = proc.stdout + proc.stderr
    return proc.returncode == 0, combined, report_path or ""


def _load_report_json(report_path: str) -> dict:
    """Load a cann-bench JSON report with proper resource mgmt."""
    with open(report_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_worst_6(cases, source_csv_dir):
    """Extract worst 6 cases from sorted case list."""
    sorted_cases = sorted(
        cases, key=lambda c: c.get("speedup", 999),
    )
    worst_6 = []
    for c in sorted_cases[:6]:
        case_id = c.get("case_id", "?")
        case_num = (
            case_id.split("_")[-1] if "_" in case_id else "?"
        )
        kernel_csv = ""
        if source_csv_dir and case_num != "?":
            csv_pattern = os.path.join(
                source_csv_dir, case_num,
                "**/kernel_details.csv",
            )
            csv_matches = glob.glob(
                csv_pattern, recursive=True,
            )
            kernel_csv = csv_matches[0] if csv_matches else ""
        worst_6.append({
            "case_id": case_id,
            "case_num": case_num,
            "baseline_us": c.get("baseline_perf_us", 0),
            "elapsed_us": c.get("elapsed_us", 0),
            "speedup": c.get("speedup", 0),
            "kernel_csv": kernel_csv,
        })
    return worst_6


def parse_cannbench_report_to_perf_result(
    report_path: str, source_csv_dir: str = "",
) -> Dict[str, Any]:
    """Parse cann-bench JSON report into perf_result."""
    if not report_path or not os.path.exists(report_path):
        return {
            "perf_pass": False,
            "overall_score": 0.0,
            "avg_speedup": 0.0,
            "_error": f"report missing: {report_path}",
        }

    data = _load_report_json(report_path)
    operators = data.get("operators", [{}])
    op = operators[0] if operators else {}
    cases = op.get("cases", [])

    overall_score = op.get("score", 0.0) or 0.0
    perf_score = op.get("performance_score", 0.0) or 0.0
    avg_speedup = op.get("avg_speedup", 0.0) or 0.0
    score_error_code = op.get("score_error_code")
    score_error = op.get("score_error")

    all_speedups = [
        c.get("speedup", 0) for c in cases
        if c.get("status") == "success"
    ]
    perf_pass = (
        len(all_speedups) > 0
        and all(s >= 1.0 for s in all_speedups)
    )

    worst_6 = _build_worst_6(cases, source_csv_dir)

    source_md = ""
    if report_path.endswith(".json"):
        source_md = report_path.replace(".json", ".md")

    return {
        "perf_pass": perf_pass,
        "overall_score": overall_score,
        "performance_score": perf_score,
        "avg_speedup": avg_speedup,
        "score_error_code": score_error_code,
        "score_error": score_error,
        "source_json": report_path,
        "source_csv_dir": source_csv_dir,
        "source_md": source_md,
        "worst_6_cases": worst_6,
    }


def parse_cannbench_report_to_precision_result(
    report_path: str,
) -> Dict[str, Any]:
    """Parse cann-bench JSON report into precision_result."""
    if not report_path or not os.path.exists(report_path):
        return {
            "precision_overall": False,
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "_error": f"report missing: {report_path}",
        }

    data = _load_report_json(report_path)
    operators = data.get("operators", [{}])
    op = operators[0] if operators else {}

    total = op.get("total_cases", 0) or 0
    passed = op.get("passed_cases", 0) or 0
    failed = op.get("failed_cases", 0) or 0

    return {
        "precision_overall": passed == total and total > 0,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "source_report": report_path,
    }
