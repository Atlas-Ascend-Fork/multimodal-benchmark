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
Agent Runner: spawns a fresh process per agent call.
Context is injected via prompt (with file paths);
agents read files themselves.
"""

import fcntl
import json
import logging
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from .handoff import read_file_safe

log = logging.getLogger("pypto-pro-workflow")

HEARTBEAT_INTERVAL = 30
DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")


# -------------------------------------------------------
# env builder
# -------------------------------------------------------

_CANN_LIBS = [
    "/home/developer/Ascend/cann-9.1.0/lib64",
    "/home/developer/Ascend/cann-9.1.0/"
    "x86_64-linux/lib64",
    "/home/developer/Ascend/cann-9.1.0/"
    "lib64/plugin/opskernel",
    "/home/developer/Ascend/cann-9.1.0/"
    "lib64/plugin/nnengine",
    "/usr/local/Ascend/driver/lib64",
    "/usr/local/Ascend/driver/lib64/common",
    "/usr/local/Ascend/driver/lib64/driver",
]


def _build_pypath(env):
    """Build PYTHONPATH with good ctypes + CANN paths."""
    good = "/usr/lib/python3.12/lib-dynload"
    cann = "/usr/local/Ascend/cann/python/site-packages"
    cann91 = (
        "/home/developer/Ascend/cann-9.1.0"
        "/python/site-packages"
    )
    tbe = (
        "/home/developer/Ascend/cann-9.1.0"
        "/opp/built-in/op_impl/ai_core/tbe"
    )
    prefix = f"{good}:" if os.path.isdir(good) else ""
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{prefix}{cann}:{cann91}:{tbe}:{old}"
    )


def _build_path(env):
    """Build PATH with CANN tool dirs."""
    cann_bin = "/home/developer/Ascend/cann-9.1.0/bin"
    ccec = (
        "/home/developer/Ascend/cann-9.1.0"
        "/tools/ccec_compiler/bin"
    )
    node_bin = (
        "/home/developer/.nvm/versions"
        "/node/v18.19.1/bin"
    )
    old = env.get("PATH", "")
    env["PATH"] = f"{cann_bin}:{ccec}:{node_bin}:{old}"


def _build_env(extra: dict = None) -> dict:
    env = os.environ.copy()
    env["TILE_FWK_DEVICE_ID"] = DEVICE_ID
    _build_pypath(env)
    _build_path(env)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        ":".join(_CANN_LIBS) + ":" + old_ld
    )
    if extra:
        env.update(extra)
    return env


# -------------------------------------------------------
# public entry
# -------------------------------------------------------

_AGENT_DISPATCH = {
    "cannbot": "_run_cannbot",
    "kerminal": "_run_kerminal",
    "hermes": "_run_hermes",
}


def run_agent(
    agent: str, role_file: str, work_dir: str,
    prompt: str = "", config: dict = None,
) -> bool:
    config = config or {}
    node_log = config.get("node_log")
    if agent == "cannbot":
        return _run_cannbot(
            role_file, work_dir, prompt, config, node_log,
        )
    if agent == "kerminal":
        return _run_kerminal(
            role_file, work_dir, prompt, config, node_log,
        )
    if agent == "hermes":
        return _run_hermes(
            role_file, work_dir, prompt, config, node_log,
        )
    log.error("Unknown agent: %s", agent)
    return False


def _run_cannbot(
    role_file, work_dir, prompt, config, node_log,
):
    """Call CANNBot in a fresh process."""
    system_prompt = read_file_safe(role_file)
    full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

    cli = config.get(
        "cli",
        "/home/developer/.nvm/versions"
        "/node/v18.19.1/bin/cannbot",
    )
    cwd = config.get("cwd", work_dir)
    cmd = [
        cli, "run", full_prompt,
        "--format", "json",
        "--dangerously-skip-permissions",
    ]

    log.info("[N1 CANNBot] start, cwd=%s", cwd)
    opts = {
        "agent_name": "N1 CANNBot",
        "node_log": node_log,
    }
    result = _run_with_heartbeat(cmd, cwd, _build_env(), opts)

    if result.returncode != 0:
        log.error(
            "[N1 CANNBot] failed, rc=%d", result.returncode,
        )
        if "Prompt exceeds max length" in result.stdout:
            log.error("[N1 CANNBot] prompt too long!")
    return result.returncode == 0


def _run_kerminal(
    role_file, work_dir, prompt, config, node_log,
):
    """Call Kerminal in a fresh process with PTY."""
    system_prompt = read_file_safe(role_file)
    full_prompt = (
        f"{system_prompt}\n\n---\n\n{prompt}"
        if prompt else system_prompt
    )

    cli = config.get("cli", "/root/.local/bin/kerminal")
    cwd = config.get("cwd", work_dir)
    cmd = [cli, full_prompt, "-a", "never", "-C", cwd]

    basename = os.path.basename(role_file)
    log.info(
        "[N2 Kerminal] start, role=%s, cwd=%s", basename, cwd,
    )
    if node_log:
        node_log.info("start, role=%s", basename)
    opts = {"node_log": node_log}
    returncode, _output = _run_with_pty(
        cmd, cwd, _build_env(), opts,
    )
    return returncode == 0


def _run_hermes(
    role_file, work_dir, prompt, config, node_log,
):
    """Call Hermes CLI in a fresh process."""
    system_prompt = read_file_safe(role_file)
    full_prompt = (
        f"{system_prompt}\n\n---\n\n{prompt}"
        if prompt else system_prompt
    )

    cli = config.get("cli", "/usr/local/bin/hermes")
    cwd = config.get("cwd", work_dir)
    cmd = [
        cli, "-z", full_prompt,
        "-t", "web,file", "--yolo", "--in", cwd,
    ]

    log.info("[N3 Hermes] start, cwd=%s", cwd)
    if node_log:
        node_log.info("start, cwd=%s", cwd)
    opts = {
        "agent_name": "N3 Hermes",
        "timeout": 0,
        "node_log": node_log,
        "plain_text": True,
    }
    result = _run_with_heartbeat(
        cmd, cwd, _build_env(), opts,
    )

    report = _find_search_report(work_dir)
    if report:
        log.info("[N3 Hermes] SEARCH_REPORT.md: %s", report)
        return True
    log.warning("[N3 Hermes] SEARCH_REPORT.md not found")
    return result.returncode == 0


def _find_search_report(work_dir: str):
    """Locate the latest SEARCH_REPORT.md."""
    search_dir = os.path.join(work_dir, "search")
    if not os.path.isdir(search_dir):
        return None
    iter_dirs = [
        d for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d))
    ]
    for d in sorted(iter_dirs, reverse=True):
        candidate = os.path.join(
            search_dir, d, "SEARCH_REPORT.md",
        )
        if os.path.exists(candidate):
            return candidate
    fallback = os.path.join(search_dir, "SEARCH_REPORT.md")
    return fallback if os.path.exists(fallback) else None


# -------------------------------------------------------
# heartbeat subprocess runner
# -------------------------------------------------------

def _parse_json_output(stripped, agent_name, nlog, state):
    """Parse one JSON line from agent stdout."""
    is_text_or_err = (
        '"type":"text"' in stripped
        or '"type":"error"' in stripped
    )
    if is_text_or_err:
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return
        if data.get("type") == "error":
            err_data = data.get("error", {})
            msg = err_data.get("data", {}).get(
                "message", str(err_data),
            )
            log.error("[%s] API error: %s", agent_name, msg)
            nlog.error("[%s] API error: %s", agent_name, msg)
            state["has_error"] = True
            state["consecutive_errors"] += 1
        elif data.get("type") == "text":
            text = data.get("part", {}).get("text", "")
            if text.strip():
                snippet = text.strip()[:200]
                log.info("[%s] out: %s", agent_name, snippet)
                nlog.info("[%s] out: %s", agent_name, snippet)
                state["consecutive_errors"] = 0
        return

    if '"tool":"' in stripped:
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return
        part = data.get("part", {})
        tool = part.get("tool", "")
        title = part.get("state", {}).get("title", "")
        status = part.get("state", {}).get("status", "")
        if tool and status == "completed":
            log.info(
                "[%s] tool:%s -> %s", agent_name, tool, title,
            )
        return

    is_error_line = (
        "error" in stripped.lower() or "ERROR" in stripped
    )
    if is_error_line:
        log.warning("[%s] %s", agent_name, stripped[:200])


def _run_with_heartbeat(cmd, cwd, env, opts):
    """Run subprocess with heartbeat and JSON parsing."""
    agent_name = opts.get("agent_name", "agent")
    timeout = opts.get("timeout", 0)
    node_log = opts.get("node_log")
    plain_text = opts.get("plain_text", False)
    nlog = node_log or log
    max_consecutive = 5

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
    )
    state = _heartbeat_loop(
        proc, nlog, agent_name, {
            "timeout": timeout,
            "plain_text": plain_text,
            "max_consecutive": max_consecutive,
        },
    )

    remaining_out, remaining_err = proc.communicate()
    if remaining_out:
        state["chunks"].append(remaining_out)
    stdout_all = "".join(state["chunks"])

    elapsed = time.time() - state["start"]
    log.info(
        "[%s] done, elapsed=%ds, rc=%d",
        agent_name, elapsed, proc.returncode,
    )
    if remaining_err:
        log.debug(
            "[%s] stderr tail:\n%s",
            agent_name, remaining_err[-500:],
        )

    _drain_plain_text(
        plain_text, remaining_out, agent_name, nlog,
    )

    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode,
        stdout=stdout_all, stderr=remaining_err or "",
    )


def _process_ready_streams(ready, ctx, state):
    """Process ready streams from select."""
    agent_name = ctx["agent_name"]
    nlog = ctx["nlog"]
    plain_text = ctx["plain_text"]
    for stream in ready:
        raw_line = stream.readline()
        if not raw_line:
            continue
        state["chunks"].append(raw_line)
        stripped = raw_line.strip()
        if not stripped:
            continue
        _parse_json_output(stripped, agent_name, nlog, state)
        if plain_text and len(stripped) > 5:
            snippet = stripped[:200]
            log.info("[%s] out: %s", agent_name, snippet)
            nlog.info("[%s] out: %s", agent_name, snippet)


def _heartbeat_loop(proc, nlog, agent_name, opts):
    """Main select loop with heartbeat."""
    timeout = opts.get("timeout", 0)
    plain_text = opts.get("plain_text", False)
    max_consecutive = opts.get("max_consecutive", 5)
    start = time.time()
    last_hb = start
    chunks = []
    state = {
        "has_error": False,
        "consecutive_errors": 0,
        "chunks": chunks,
        "start": start,
    }
    ctx = {
        "agent_name": agent_name,
        "nlog": nlog,
        "plain_text": plain_text,
    }

    while proc.poll() is None:
        elapsed = time.time() - start
        if timeout > 0 and elapsed > timeout:
            proc.kill()
            log.error("[%s] timeout (%ds)", agent_name, timeout)
            break

        errs = state["consecutive_errors"]
        if errs >= max_consecutive:
            proc.kill()
            msg = f"[{agent_name}] {errs} consecutive errors"
            log.error(msg)
            nlog.error(msg)
            break

        ready, _, _ = select.select(
            [proc.stdout, proc.stderr], [], [], 1.0,
        )
        _process_ready_streams(ready, ctx, state)

        if not ready:
            now = time.time()
            if now - last_hb >= HEARTBEAT_INTERVAL:
                log.info(
                    "[%s] running... (%ds)",
                    agent_name, now - start,
                )
                last_hb = now

    return state


def _drain_plain_text(plain_text, remaining, name, nlog):
    """Log remaining plain-text output after process ends."""
    if not plain_text or not remaining:
        return
    remaining = remaining.strip()
    if not remaining:
        return
    for out_line in remaining.split("\n"):
        out_line = out_line.strip()
        if out_line and len(out_line) > 5:
            log.info("[%s] out: %s", name, out_line[:200])
            nlog.info("[%s] out: %s", name, out_line[:200])


# -------------------------------------------------------
# PTY runner (for kerminal)
# -------------------------------------------------------

_ANSI_RE = re.compile(
    r'\x1b\[[0-9;]*[A-Za-z]'
    r'|\x1b\].*?\x07'
    r'|\x1b[()][0-9A-B]'
    r'|\x1b[M]|\x0f|\x0e|\r'
    r'|\x1b\[\?[0-9]*[hl]'
)

_TUI_BORDER_CHARS = frozenset("╭│╰├")


def _extract_tui_lines(raw):
    """Extract readable lines from TUI raw output."""
    clean = _ANSI_RE.sub("", raw)
    clean = clean.replace("\u2022", "").replace("\u203a", "")
    lines = []
    for ln in clean.split("\n"):
        ln = ln.strip()
        is_long_enough = len(ln) > 10
        is_not_border = ln[:1] not in _TUI_BORDER_CHARS
        if is_long_enough and is_not_border:
            lines.append(ln)
    return lines


def _run_with_pty(cmd, cwd, env, opts):
    """Run a command that needs a TTY (kerminal)."""
    node_log = opts.get("node_log")
    timeout = opts.get("timeout", 0)
    agent_label = opts.get("agent_label", "N2 Kerminal")
    nlog = node_log or log

    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", 50, 200, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    run_env = env.copy()
    run_env["TERM"] = "xterm-256color"
    run_env["COLUMNS"] = "200"
    run_env["LINES"] = "50"

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdin=slave_fd,
        stdout=slave_fd, stderr=slave_fd,
        env=run_env, close_fds=True,
    )
    os.close(slave_fd)

    pty_state = {
        "chunks": [],
        "tui_buf": "",
        "logged": set(),
        "start": time.time(),
        "last_data": time.time(),
        "last_hb": time.time(),
        "enter_sent": False,
        "task_started": False,
    }

    try:
        _pty_select_loop(
            proc, master_fd, nlog,
            {"label": agent_label, "timeout": timeout},
            pty_state,
        )
    finally:
        os.close(master_fd)

    proc.wait()
    elapsed = time.time() - pty_state["start"]
    log.info(
        "[%s] done, elapsed=%ds, rc=%d",
        agent_label, elapsed, proc.returncode,
    )
    nlog.info("done, elapsed=%ds, rc=%d", elapsed, proc.returncode)
    return proc.returncode, "".join(pty_state["chunks"])


def _pty_select_loop(
    proc, master_fd, nlog, label_and_opts, state,
):
    """PTY select loop with idle detection."""
    agent_label = label_and_opts.get("label", "N2 Kerminal")
    timeout = label_and_opts.get("timeout", 0)
    idle_seconds = 15

    while True:
        elapsed = time.time() - state["start"]
        if timeout > 0 and elapsed > timeout:
            proc.kill()
            break

        ready, _, _ = select.select(
            [master_fd], [], [], 1.0,
        )
        if ready:
            _pty_handle_data(proc, master_fd, state)
        else:
            _pty_handle_idle(
                proc, nlog, agent_label, state, idle_seconds,
            )

        if proc.poll() is not None:
            _pty_drain(master_fd, state)
            break


def _pty_handle_data(proc, master_fd, state):
    """Read data from PTY master fd."""
    try:
        data = os.read(master_fd, 8192)
    except OSError:
        proc.kill()
        return
    if not data:
        proc.kill()
        return

    text = data.decode("utf-8", errors="replace")
    state["chunks"].append(text)
    state["last_data"] = time.time()
    state["tui_buf"] += text

    if "\x1b[6n" in text:
        os.write(master_fd, b"\x1b[1;1R")
    if not state["enter_sent"]:
        if "Press Enter to continue" in text:
            os.write(master_fd, b"\r")
            state["enter_sent"] = True
    if "Worked for" in text or "context left" in text:
        state["task_started"] = True


def _pty_handle_idle(
    proc, nlog, agent_label, state, idle_seconds,
):
    """Handle idle timeout and TUI buffer flushing."""
    if state["tui_buf"]:
        lines = _extract_tui_lines(state["tui_buf"])
        skip = {"Worked for", "context left", "Press Enter"}
        for ln in lines:
            should_skip = any(s in ln for s in skip)
            if ln not in state["logged"] and not should_skip:
                nlog.info("out: %s", ln[:200])
                log.info("[%s] out: %s", agent_label, ln[:200])
                state["logged"].add(ln)
        state["tui_buf"] = ""

    now = time.time()
    if now - state["last_hb"] >= HEARTBEAT_INTERVAL:
        elapsed = now - state["start"]
        log.info("[%s] running... (%ds)", agent_label, elapsed)
        nlog.info("running... (%ds)", elapsed)
        state["last_hb"] = now

    idle = now - state["last_data"]
    if state["task_started"] and idle > idle_seconds:
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()


def _pty_drain(master_fd, state):
    """Drain remaining PTY output after process exits."""
    while True:
        ready, _, _ = select.select(
            [master_fd], [], [], 0.1,
        )
        if not ready:
            break
        try:
            data = os.read(master_fd, 8192)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            state["chunks"].append(text)
        except OSError:
            break
