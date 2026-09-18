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
History Manager: cross-iteration context memory.
Storage: <work>/knowledge/history.json
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def _history_path(work_dir: str) -> str:
    return os.path.join(work_dir, "knowledge", "history.json")


_EMPTY_HISTORY = {
    "insights": [],
    "ledger": [],
    "rounds": [],
    "bottleneck_now": "",
    "suggest_next": [],
    "worst_cases_tracker": {},
    "fusion_kernel_strategy": [],
}


def load_history(work_dir: str) -> Dict[str, Any]:
    path = _history_path(work_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return dict(_EMPTY_HISTORY)


def save_history(work_dir: str, history: Dict[str, Any]):
    path = _history_path(work_dir)
    Path(os.path.dirname(path)).mkdir(
        parents=True, exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)


def _classify_verdict(after: float, before: float) -> str:
    if before <= 0:
        return "baseline"
    delta_pct = (after - before) / before * 100
    if delta_pct >= 5:
        return "big_win"
    if delta_pct >= 1:
        return "small_win"
    if delta_pct >= -1:
        return "no_change"
    return "regression"


def _build_round_entry(iteration, perf_result, prev_avg):
    """Build a rounds[] entry from perf data."""
    avg = perf_result.get("avg_speedup", 0.0)
    overall = perf_result.get("overall_score", 0.0)
    perf_sc = perf_result.get("performance_score", 0.0)
    cases = (
        perf_result.get("cases")
        or perf_result.get("worst_6_cases", [])
    )
    sorted_c = sorted(
        cases, key=lambda c: c.get("speedup", 999),
    )
    worst_6 = sorted_c[:6]
    improved = avg > prev_avg if prev_avg > 0 else True

    w_id = worst_6[0].get("case_id", "?") if worst_6 else "?"
    w_spd = worst_6[0].get("speedup", 0) if worst_6 else 0

    return {
        "iter": iteration,
        "avg_speedup": round(avg, 4),
        "overall_score": round(overall, 2),
        "performance_score": round(perf_sc, 2),
        "improved": improved,
        "worst_case": {
            "case_id": w_id,
            "speedup": round(w_spd, 4),
        },
        "worst_6_speedups": [
            {
                "case_id": c.get("case_id", "?"),
                "speedup": round(c.get("speedup", 0), 4),
            }
            for c in worst_6
        ],
    }


def _build_ledger_entry(iteration, perf_result, prev_avg):
    """Build a ledger[] entry from perf data."""
    avg = perf_result.get("avg_speedup", 0.0)
    return {
        "iter": iteration,
        "reason": "(pending tech_lead)",
        "direction": "(pending tech_lead)",
        "modify_files": [],
        "readonly_files": [],
        "avg_speedup_before": round(prev_avg, 4),
        "avg_speedup_after": round(avg, 4),
        "delta": round(avg - prev_avg, 4),
        "verdict": _classify_verdict(avg, prev_avg),
    }


def append_round(
    work_dir: str, iteration: int,
    perf_result: Dict[str, Any],
    prev_avg_speedup: float = 0.0,
) -> Dict[str, Any]:
    """Append hard data for this round."""
    history = load_history(work_dir)

    history["rounds"] = [
        r for r in history.get("rounds", [])
        if r.get("iter") != iteration
    ]
    history["ledger"] = [
        e for e in history.get("ledger", [])
        if e.get("iter") != iteration
    ]

    round_entry = _build_round_entry(
        iteration, perf_result, prev_avg_speedup,
    )
    history["rounds"].append(round_entry)

    ledger_entry = _build_ledger_entry(
        iteration, perf_result, prev_avg_speedup,
    )
    history["ledger"].append(ledger_entry)

    save_history(work_dir, history)
    _save_rounds_snapshot(work_dir, history["rounds"])
    return history


def append_ledger_only(
    work_dir: str, iteration: int, fail_reason: str,
) -> Dict[str, Any]:
    """Append ledger entry when no perf data available."""
    history = load_history(work_dir)
    history["ledger"] = [
        e for e in history.get("ledger", [])
        if e.get("iter") != iteration
    ]
    ledger_entry = {
        "iter": iteration,
        "reason": fail_reason,
        "direction": "(pending tech_lead)",
        "modify_files": [],
        "readonly_files": [],
        "avg_speedup_before": None,
        "avg_speedup_after": None,
        "delta": None,
        "verdict": fail_reason,
    }
    history["ledger"].append(ledger_entry)
    save_history(work_dir, history)
    return history


# PLACEHOLDER_UPDATE_FROM_TECH_LEAD


def _apply_insights_and_bottleneck(history, output):
    """Apply insights, bottleneck, suggest_next, tracker."""
    if "insights" in output:
        history["insights"] = output["insights"]
    if "bottleneck_now" in output:
        history["bottleneck_now"] = output["bottleneck_now"]
    if "suggest_next" in output:
        history["suggest_next"] = output["suggest_next"]
    if "worst_cases_tracker" in output:
        history["worst_cases_tracker"] = (
            output["worst_cases_tracker"]
        )


def _apply_ledger_direction(history, output, iteration):
    """Fill in direction for this iteration's ledger."""
    if "direction_for_ledger" not in output:
        return
    if not history["ledger"]:
        return
    for entry in reversed(history["ledger"]):
        if entry.get("iter") == iteration:
            entry["direction"] = (
                output["direction_for_ledger"]
            )
            break


def _apply_fusion_strategy(history, output):
    """Append new fusion_kernel_strategy entries."""
    new_fds = output.get("fusion_kernel_strategy")
    if new_fds is None:
        return
    existing = history.get("fusion_kernel_strategy", [])
    if isinstance(new_fds, dict):
        new_fds = [new_fds]
    if isinstance(new_fds, list):
        for fd in new_fds:
            if isinstance(fd, dict) and fd not in existing:
                existing.append(fd)
    history["fusion_kernel_strategy"] = existing


def _apply_pending_fields(history, output):
    """Apply exit_decision, proven/regression patterns."""
    if "exit_decision" in output:
        history["exit_decision"] = output["exit_decision"]
    if "proven_pattern" in output:
        history["_pending_proven_pattern"] = (
            output["proven_pattern"]
        )
    if "regression_pattern" in output:
        history["_pending_regression_pattern"] = (
            output["regression_pattern"]
        )
    if "pitfall" in output:
        history["_pending_pitfall"] = output["pitfall"]


def update_from_tech_lead(
    work_dir: str,
    tech_lead_output: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    """Update history from tech_lead output."""
    history = load_history(work_dir)
    _apply_insights_and_bottleneck(history, tech_lead_output)
    _apply_ledger_direction(
        history, tech_lead_output, iteration,
    )
    _apply_fusion_strategy(history, tech_lead_output)
    _apply_pending_fields(history, tech_lead_output)
    save_history(work_dir, history)
    return history


# -------------------------------------------------------
# proven patterns
# -------------------------------------------------------

def _ensure_file_with_header(path, title, description):
    """Create a pattern file with header if missing."""
    Path(os.path.dirname(path)).mkdir(
        parents=True, exist_ok=True,
    )
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {title}\n\n")
            fh.write(f"> {description}\n\n")


def append_proven_pattern(
    work_dir: str, iteration: int,
    pattern: Dict[str, Any],
):
    """Append a proven optimization pattern."""
    path = os.path.join(
        work_dir, "knowledge", "proven_patterns.md",
    )
    _ensure_file_with_header(
        path,
        "Proven optimization patterns",
        "Each entry records a big speedup (>=5%).",
    )
    fusion_tag = (
        " fusion-related" if pattern.get("fusion_related")
        else ""
    )
    before = pattern.get("speedup_before", "?")
    after = pattern.get("speedup_after", "?")
    delta = pattern.get("delta_pct", "?")
    what = pattern.get("what_changed", "?")
    why = pattern.get("why_it_worked", "?")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            f"## iter{iteration}: speedup {before}"
            f" -> {after} (+{delta}%){fusion_tag}\n\n"
        )
        fh.write(f"- **what_changed**: {what}\n")
        fh.write(f"- **why_it_worked**: {why}\n\n")


def _parse_pattern_blocks(content, header_re):
    """Parse pattern blocks from markdown content."""
    blocks = re.split(
        r"(?=^## iter)", content, flags=re.MULTILINE,
    )
    patterns = []
    for blk in blocks:
        blk = blk.strip()
        if not blk.startswith("## iter"):
            continue
        header = blk.split("\n")[0]
        match = header_re.match(header)
        if not match:
            continue
        patterns.append((match, blk))
    return patterns


_PROVEN_RE = re.compile(
    r"## iter(\d+).*?speedup\s+([\d.?]+)"
    r"\s*(?:→|->)\s*([\d.?]+)\s*"
    r"\(\+([\d.?]+)%\)(.*)"
)

_REGRESSION_RE = re.compile(
    r"## iter(\d+).*?speedup\s+([\d.?]+)"
    r"\s*(?:→|->)\s*([\d.?]+)\s*"
    r"\(([-\d.?]+)%\)(.*)"
)


def _safe_float(val, default=0.0):
    return float(val) if val != "?" else default


def _parse_pattern_body_fields(blk, field_map):
    """Extract named fields from a pattern markdown block."""
    extracted = {}
    for ln in blk.split("\n"):
        for prefix, field_name in field_map.items():
            if ln.startswith(prefix):
                extracted[field_name] = ln.split(":", 1)[1].strip()
                break
    return extracted


_PROVEN_FIELD_MAP = {
    "- **what_changed**:": "what_changed",
    "- **改了什么**:": "what_changed",
    "- **why_it_worked**:": "why_it_worked",
    "- **为什么有效**:": "why_it_worked",
}

_REGRESSION_FIELD_MAP = {
    "- **what_changed**:": "what_changed",
    "- **改了什么**:": "what_changed",
    "- **why_it_failed**:": "why_it_failed",
    "- **为什么退步**:": "why_it_failed",
}


def load_proven_patterns(work_dir: str) -> List[Dict[str, Any]]:
    """Read accumulated proven patterns."""
    path = os.path.join(
        work_dir, "knowledge", "proven_patterns.md",
    )
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    results = []
    for match, blk in _parse_pattern_blocks(content, _PROVEN_RE):
        entry = {
            "iter": int(match.group(1)),
            "speedup_before": _safe_float(match.group(2)),
            "speedup_after": _safe_float(match.group(3)),
            "delta_pct": _safe_float(match.group(4)),
            "fusion_related": "fusion" in match.group(5).lower()
            or "融合" in match.group(5),
            "what_changed": "",
            "why_it_worked": "",
        }
        entry.update(_parse_pattern_body_fields(blk, _PROVEN_FIELD_MAP))
        results.append(entry)
    return results


# PLACEHOLDER_REST


def format_proven_patterns_for_prompt(
    work_dir: str,
) -> str:
    """Format proven patterns for prompt injection."""
    patterns = load_proven_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\nProven optimization patterns:",
        "These record big speedups (>=5%).",
        "",
    ]
    for p in patterns:
        fusion = " [fusion]" if p.get("fusion_related") else ""
        before = p.get("speedup_before", "?")
        after = p.get("speedup_after", "?")
        delta = p.get("delta_pct", "?")
        lines.append(
            f"  iter{p.get('iter', '?')}: "
            f"{before} -> {after} (+{delta}%){fusion}"
        )
        lines.append(
            f"     changed: {p.get('what_changed', '?')}"
        )
        lines.append(
            f"     reason: {p.get('why_it_worked', '?')}"
        )
        lines.append("")
    return "\n".join(lines)


def append_regression_pattern(
    work_dir: str, iteration: int,
    pattern: Dict[str, Any],
):
    """Append a regression pattern."""
    path = os.path.join(
        work_dir, "knowledge", "regression_patterns.md",
    )
    _ensure_file_with_header(
        path,
        "Regression patterns",
        "Each entry records a big regression (>=5%).",
    )
    fusion_tag = (
        " fusion-related" if pattern.get("fusion_related")
        else ""
    )
    before = pattern.get("speedup_before", "?")
    after = pattern.get("speedup_after", "?")
    delta = pattern.get("delta_pct", "?")
    what = pattern.get("what_changed", "?")
    why = pattern.get("why_it_failed", "?")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            f"## iter{iteration}: speedup {before}"
            f" -> {after} ({delta}%){fusion_tag}\n\n"
        )
        fh.write(f"- **what_changed**: {what}\n")
        fh.write(f"- **why_it_failed**: {why}\n\n")


def load_regression_patterns(
    work_dir: str,
) -> List[Dict[str, Any]]:
    """Read accumulated regression patterns."""
    path = os.path.join(
        work_dir, "knowledge", "regression_patterns.md",
    )
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    results = []
    for match, blk in _parse_pattern_blocks(
        content, _REGRESSION_RE,
    ):
        entry = {
            "iter": int(match.group(1)),
            "speedup_before": _safe_float(match.group(2)),
            "speedup_after": _safe_float(match.group(3)),
            "delta_pct": _safe_float(match.group(4)),
            "fusion_related": (
                "fusion" in match.group(5).lower()
                or "融合" in match.group(5)
            ),
            "what_changed": "",
            "why_it_failed": "",
        }
        entry.update(_parse_pattern_body_fields(blk, _REGRESSION_FIELD_MAP))
        results.append(entry)
    return results


def format_regression_patterns_for_prompt(
    work_dir: str,
) -> str:
    """Format regression patterns for prompt injection."""
    patterns = load_regression_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\nRegression patterns (AVOID these):",
        "",
    ]
    for p in patterns:
        fusion = " [fusion]" if p.get("fusion_related") else ""
        before = p.get("speedup_before", "?")
        after = p.get("speedup_after", "?")
        delta = p.get("delta_pct", "?")
        lines.append(
            f"  iter{p.get('iter', '?')}: "
            f"{before} -> {after} ({delta}%){fusion}"
        )
        lines.append(
            f"     changed: {p.get('what_changed', '?')}"
        )
        lines.append(
            f"     failed: {p.get('why_it_failed', '?')}"
        )
        lines.append("")
    return "\n".join(lines)



def append_pitfall(
    work_dir: str, iteration: int,
    pitfall: Dict[str, Any],
):
    """Append a tech_lead pitfall entry."""
    path = os.path.join(
        work_dir, "knowledge", "tech_lead_pitfalls.md",
    )
    _ensure_file_with_header(
        path,
        "Tech Lead pitfalls",
        "Records tech_lead misjudgments.",
    )
    verdict = pitfall.get("verdict", "confirmed")
    mark = "confirmed" if verdict == "confirmed" else "rejected"
    topic = pitfall.get("topic", "?")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            f"## iter{iteration} {mark}: {topic}\n\n"
        )
        fh.write(
            f"- **target_advice**: "
            f"{pitfall.get('target_advice', '?')}\n"
        )
        fh.write(
            f"- **feedback**: "
            f"{pitfall.get('feedback', '?')}\n"
        )
        fh.write(
            f"- **root_cause**: "
            f"{pitfall.get('root_cause', '?')}\n"
        )
        fh.write(
            f"- **correct_approach**: "
            f"{pitfall.get('correct_approach', '?')}\n\n"
        )


_PITFALL_RE = re.compile(
    r"## iter(\d+)\s+(confirmed|rejected"
    r"|✅确认误判|❌驳回):\s*(.*)"
)


def load_pitfalls(work_dir: str) -> List[Dict[str, Any]]:
    """Read the pitfalls notebook."""
    path = os.path.join(
        work_dir, "knowledge", "tech_lead_pitfalls.md",
    )
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    pitfalls = []
    blocks = re.split(
        r"(?=^## iter)", content, flags=re.MULTILINE,
    )
    for blk in blocks:
        blk = blk.strip()
        if not blk.startswith("## iter"):
            continue
        header = blk.split("\n")[0]
        match = _PITFALL_RE.match(header)
        if not match:
            continue
        is_confirmed = "confirmed" in match.group(2)
        is_confirmed = is_confirmed or "确认" in match.group(2)
        entry = {
            "iter": int(match.group(1)),
            "verdict": "confirmed" if is_confirmed else "rejected",
            "topic": match.group(3).strip(),
            "target_advice": "",
            "feedback": "",
            "root_cause": "",
            "correct_approach": "",
        }
        _parse_pitfall_body(entry, blk)
        pitfalls.append(entry)
    return pitfalls


def _parse_pitfall_body(entry, blk):
    """Parse body fields from a pitfall block."""
    field_map = {
        "- **target_advice**:": "target_advice",
        "- **涉及的意见**:": "target_advice",
        "- **feedback**:": "feedback",
        "- **cannbot 反馈**:": "feedback",
        "- **root_cause**:": "root_cause",
        "- **确认误判原因**:": "root_cause",
        "- **驳回原因**:": "root_cause",
        "- **correct_approach**:": "correct_approach",
        "- **正确做法**:": "correct_approach",
        "- **正确认知**:": "correct_approach",
    }
    for ln in blk.split("\n"):
        for prefix, field_name in field_map.items():
            if ln.startswith(prefix):
                entry[field_name] = ln.split(":", 1)[1].strip()
                break


def format_pitfalls_for_prompt(work_dir: str) -> str:
    """Format pitfalls for prompt injection."""
    pitfalls = load_pitfalls(work_dir)
    if not pitfalls:
        return ""
    lines = [
        "\nTech Lead pitfalls notebook:",
        "",
    ]
    for p in pitfalls:
        mark = "confirmed" if p["verdict"] == "confirmed" else "rejected"
        lines.append(
            f"  [{mark}] iter{p['iter']}: {p['topic']}"
        )
        lines.append(
            f"     advice: {p['target_advice']}"
        )
        lines.append(
            f"     correct: {p['correct_approach']}"
        )
        lines.append("")
    return "\n".join(lines)


# PLACEHOLDER_FORMAT_FOR_PROMPT


def _fmt_suggest_item(item):
    """Format a single suggest_next item dict into a line."""
    pri = item.get("priority", "P2")
    action = item.get("action", "")
    reason = item.get("reason", "")
    marker = {
        "P0": "[!]", "P1": "[+]", "P2": "[.]",
    }.get(pri, "[?]")
    line = f"- {marker} [{pri}] {action}"
    if reason:
        line += f" ({reason})"
    return line


def _fmt_suggest_next(history, parts):
    """Format suggest_next section."""
    suggest = history.get("suggest_next", [])
    if not suggest:
        return
    parts.append(
        "## Modification instructions (by priority)\n"
    )
    if isinstance(suggest, list):
        for item in suggest:
            if isinstance(item, dict):
                parts.append(_fmt_suggest_item(item))
            else:
                parts.append(f"- {item}")
    elif isinstance(suggest, str) and suggest.strip():
        parts.append(f"- [!] [P0] {suggest}")


def _fmt_insights(history, parts):
    """Format insights section."""
    if not history.get("insights"):
        return
    parts.append("\n## Insights knowledge base\n")
    for i, insight in enumerate(history["insights"], 1):
        parts.append(f"{i}. {insight}")


def _fmt_bottleneck(history, parts):
    """Format bottleneck section."""
    if not history.get("bottleneck_now"):
        return
    parts.append("\n## Current bottleneck\n")
    parts.append(history["bottleneck_now"])


def _fmt_worst_tracker(history, parts):
    """Format worst_cases_tracker section."""
    tracker = history.get("worst_cases_tracker")
    if not tracker:
        return
    parts.append("\n## Worst case tracker\n")
    if isinstance(tracker, dict):
        for case_id, info in tracker.items():
            parts.append(f"- case_{case_id}: {info}")
    elif isinstance(tracker, list):
        for item in tracker:
            parts.append(f"- {item}")


def _fmt_fusion_strategy(history, parts):
    """Format fusion_kernel_strategy section."""
    fds = history.get("fusion_kernel_strategy", [])
    if not fds:
        return
    parts.append("\n## Fusion kernel strategy\n")
    for fd in fds:
        if isinstance(fd, dict):
            it = fd.get("iter", "?")
            st = fd.get("status", "?")
            dr = fd.get("direction", "?")
            ev = fd.get("evidence", "none")
            parts.append(
                f"- iter{it}: [{st}] {dr}\n"
                f"  evidence: {ev}"
            )
        else:
            parts.append(f"- {fd}")


def _fmt_patterns(work_dir, parts):
    """Format proven + regression patterns."""
    proven = load_proven_patterns(work_dir)
    if proven:
        parts.append("\n## Proven patterns\n")
        for p in proven:
            ftag = " [fusion]" if p.get("fusion_related") else ""
            before = p.get("speedup_before", "?")
            after = p.get("speedup_after", "?")
            delta = p.get("delta_pct", "?")
            parts.append(
                f"- iter{p.get('iter', '?')}: "
                f"{before} -> {after} (+{delta}%){ftag}\n"
                f"  changed: {p.get('what_changed', '?')}\n"
                f"  reason: {p.get('why_it_worked', '?')}"
            )

    regressions = load_regression_patterns(work_dir)
    if regressions:
        parts.append("\n## Regression patterns (AVOID)\n")
        for p in regressions:
            ftag = " [fusion]" if p.get("fusion_related") else ""
            before = p.get("speedup_before", "?")
            after = p.get("speedup_after", "?")
            delta = p.get("delta_pct", "?")
            parts.append(
                f"- iter{p.get('iter', '?')}: "
                f"{before} -> {after} ({delta}%){ftag}\n"
                f"  changed: {p.get('what_changed', '?')}\n"
                f"  failed: {p.get('why_it_failed', '?')}"
            )


def _fmt_ledger(history, parts, max_ledger):
    """Format ledger section."""
    ledger = history.get("ledger", [])
    if not ledger:
        return
    recent = ledger[-max_ledger:]
    parts.append(
        f"\n## Iteration ledger (last {len(recent)})\n"
    )
    for entry in recent:
        reason = entry.get("reason", "?")
        direction = entry.get("direction", "?")
        modify = entry.get("modify_files", [])
        readonly = entry.get("readonly_files", [])
        verdict = entry.get("verdict", "?")
        spd_before = entry.get("avg_speedup_before", "?")
        spd_after = entry.get("avg_speedup_after", "?")

        parts.append(
            f"### iter{entry['iter']} ({reason}) -> {verdict}"
        )
        parts.append(f"direction: {direction}")
        parts.append(f"speedup: {spd_before} -> {spd_after}")
        if modify:
            parts.append(f"modify: {', '.join(modify)}")
        if readonly:
            tail = "..." if len(readonly) > 5 else ""
            parts.append(
                f"readonly: {', '.join(readonly[:5])}{tail}"
            )
        parts.append("")


def _fmt_rounds(history, parts):
    """Format rounds trend section."""
    rounds = history.get("rounds", [])
    if not rounds:
        return
    parts.append(f"\n## Performance trend ({len(rounds)} rounds)\n")
    for r in rounds:
        mark = "up" if r.get("improved") else "down"
        wc = r.get("worst_case", {})
        wc_id = wc.get("case_id", "?")
        wc_spd = wc.get("speedup", "?")
        parts.append(
            f"- iter{r['iter']}: "
            f"avg={r['avg_speedup']} [{mark}] "
            f"| worst=case_{wc_id}({wc_spd})"
        )


def format_for_prompt(
    work_dir: str, max_ledger: int = 10,
) -> str:
    """Format history for stage3 prompt injection."""
    history = load_history(work_dir)
    parts = []

    _fmt_suggest_next(history, parts)
    _fmt_insights(history, parts)
    _fmt_bottleneck(history, parts)
    _fmt_worst_tracker(history, parts)
    _fmt_fusion_strategy(history, parts)
    _fmt_patterns(work_dir, parts)
    _fmt_ledger(history, parts, max_ledger)
    _fmt_rounds(history, parts)

    return "\n".join(parts) if parts else "(no history)"


def get_latest_fix_plan(work_dir: str) -> str:
    """Extract file constraints from last ledger entry."""
    history = load_history(work_dir)
    ledger = history.get("ledger", [])
    if not ledger:
        return ""

    latest = ledger[-1]
    direction = latest.get("direction", "")
    modify = latest.get("modify_files", [])
    readonly = latest.get("readonly_files", [])
    reason = latest.get("reason", "")

    is_pending = direction == "(pending tech_lead)"
    if is_pending or (not modify and not readonly):
        return ""

    lines = [
        "=== Tech Lead modification plan ===",
        f"issue: {reason}",
        f"direction: {direction}",
    ]
    if modify:
        lines.append("modifiable files:")
        for mf in modify:
            lines.append(f"  - {mf}")
    if readonly:
        lines.append("read-only files:")
        for rf in readonly:
            lines.append(f"  - {rf}")
    lines.append("Only modify files in the modify list.")
    return "\n".join(lines)


# -------------------------------------------------------
# rounds snapshot protection
# -------------------------------------------------------

def _rounds_snapshot_path(work_dir: str) -> str:
    return os.path.join(
        work_dir, "knowledge", "_rounds_snapshot.json",
    )


def _fusion_snapshot_path(work_dir: str) -> str:
    return os.path.join(
        work_dir, "knowledge", "_fusion_snapshot.json",
    )


def _save_rounds_snapshot(
    work_dir: str, rounds: List[Dict[str, Any]],
):
    path = _rounds_snapshot_path(work_dir)
    Path(os.path.dirname(path)).mkdir(
        parents=True, exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rounds, fh, ensure_ascii=False, indent=2)


def save_pre_tech_lead_snapshot(work_dir: str):
    """Backup rounds and fusion strategy before stage9."""
    history = load_history(work_dir)
    _save_rounds_snapshot(
        work_dir, history.get("rounds", []),
    )
    fd_path = _fusion_snapshot_path(work_dir)
    Path(os.path.dirname(fd_path)).mkdir(
        parents=True, exist_ok=True,
    )
    with open(fd_path, "w", encoding="utf-8") as fh:
        json.dump(
            history.get("fusion_kernel_strategy", []),
            fh, ensure_ascii=False, indent=2,
        )


def _restore_rounds(history, work_dir):
    """Restore rounds from snapshot."""
    snap = _rounds_snapshot_path(work_dir)
    if os.path.exists(snap):
        with open(snap, "r", encoding="utf-8") as fh:
            history["rounds"] = json.load(fh)


def _merge_fusion_snapshot(history, work_dir):
    """Merge fusion strategy with snapshot (keep old + new)."""
    snap = _fusion_snapshot_path(work_dir)
    if not os.path.exists(snap):
        return
    with open(snap, "r", encoding="utf-8") as fh:
        old_fds = json.load(fh)
    current = history.get("fusion_kernel_strategy", [])
    merged = list(old_fds)
    for fd in current:
        if fd not in merged:
            merged.append(fd)
    history["fusion_kernel_strategy"] = merged


def restore_program_fields(
    work_dir: str,
) -> Dict[str, Any]:
    """Restore program-protected fields after stage9."""
    history = load_history(work_dir)
    _restore_rounds(history, work_dir)
    _merge_fusion_snapshot(history, work_dir)
    save_history(work_dir, history)
    return history
