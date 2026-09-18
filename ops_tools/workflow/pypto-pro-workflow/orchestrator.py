# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PyPTO Pro Operator Workflow — 确定性控制平面
========================================
唯一启动入口。纯 Python if/else + while 循环，零 LLM 调用。
阶段切换完全由本程序控制，agent 只执行不判断流程。

用法:
    python orchestrator.py --task-dir /path/to/cannbench/task/<op>
    python orchestrator.py --task-dir /path/to/task --max-iter 20
"""

import argparse
import atexit
import datetime
import glob
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from lib.agent_runner import run_agent
from lib.bench_parser import (
    analyze_kernel_csv_for_anticheat,
    check_build_success,
    parse_cannbench_report_to_perf_result,
    parse_cannbench_report_to_precision_result,
    parse_perf_result,
    parse_precision_result,
    run_perf_eval,
    run_precision_eval,
)
from lib.handoff import atomic_write_json, read_file_safe
from lib.history_manager import (
    append_ledger_only,
    append_pitfall,
    append_proven_pattern,
    append_regression_pattern,
    append_round,
    format_for_prompt,
    format_pitfalls_for_prompt,
    format_proven_patterns_for_prompt,
    format_regression_patterns_for_prompt,
    get_latest_fix_plan,
    load_history,
    load_pitfalls,
    load_proven_patterns,
    load_regression_patterns,
    restore_program_fields,
    save_history,
    save_pre_tech_lead_snapshot,
)
from lib.logger import (
    setup_history_logger,
    setup_logger,
    setup_node_logger,
    setup_state_logger,
)
from lib.state import State

logger = logging.getLogger(__name__)


class _WfCtx:
    """Bundles workflow-wide state to reduce parameter counts."""


# ── NPU device detection helpers ─────────────────────────

_DETECT_CHIP_SCRIPT = (
    "import torch, torch_npu\n"
    "props = torch_npu.npu.get_device_properties(0)\n"
    "print(props.name)\n"
    "print(props.cube_core_num)\n"
    "print(props.vector_core_num)\n"
    "print(props.L2_cache_size)\n"
)


def _bash_run_script(script_path):
    """Run a Python script via bash with CANN env setup."""
    cmd = [
        "bash", "-c",
        "source ~/.bashrc 2>/dev/null; "
        "export PYTHONPATH="
        "/usr/lib/python3.12/lib-dynload:$PYTHONPATH; "
        "unset LD_PRELOAD; "
        "python3 " + script_path,
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=30
    )


def _parse_chip_props(info, lines):
    """Parse optional chip properties from detect output."""
    field_map = [(1, "cube_core_num"), (2, "vector_core_num")]
    for idx, key in field_map:
        if len(lines) > idx:
            try:
                info[key] = int(lines[idx])
            except ValueError:
                pass
    if len(lines) > 3:
        try:
            info["l2_size_kb"] = int(lines[3]) // 1024
        except ValueError:
            pass


def _detect_soc_via_torch(info):
    """Detect SoC via torch_npu with up to 3 retries."""
    script_path = "/tmp/_pypto_detect_chip.py"
    with open(script_path, "w", encoding="utf-8") as fobj:
        fobj.write(_DETECT_CHIP_SCRIPT)
    for attempt in range(1, 4):
        try:
            result = _bash_run_script(script_path)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "[芯片检测] 第%d次异常: %s", attempt, exc
            )
            if attempt < 3:
                time.sleep(5)
            continue
        if result.returncode != 0:
            tail = result.stderr[-300:] if result.stderr else "(空)"
            logger.warning(
                "[芯片检测] 第%d次失败(rc=%d): %s",
                attempt, result.returncode, tail,
            )
            if attempt < 3:
                logger.info("[芯片检测] 等待5秒后重试...")
                time.sleep(5)
            continue
        lines = result.stdout.strip().split("\n")
        if not (lines and lines[0].strip()):
            continue
        info["soc_version"] = lines[0].strip()
        _parse_chip_props(info, lines)
        return lines[0].strip()
    return None


def _detect_soc_from_env():
    """Try SoC name from environment variables."""
    for key in ("SOC_VERSION", "ASCEND_SOC_VERSION"):
        val = os.environ.get(key, "")
        if val:
            return val
    return None


def _query_tbe_params(soc_name, info):
    """Query on-chip memory parameters via tbe API."""
    script = "/tmp/_pypto_detect_tbe.py"
    with open(script, "w", encoding="utf-8") as fobj:
        fobj.write(
            "from tbe.common.platform import "
            "get_soc_spec, set_current_compile_soc_info\n"
            "set_current_compile_soc_info('" + soc_name + "')\n"
            "print(get_soc_spec('UB_SIZE'))\n"
            "print(get_soc_spec('L1_SIZE'))\n"
            "print(get_soc_spec('L0A_SIZE'))\n"
            "print(get_soc_spec('L0B_SIZE'))\n"
            "print(get_soc_spec('L0C_SIZE'))\n"
            "print(get_soc_spec('CORE_NUM'))\n"
        )
    try:
        result = _bash_run_script(script)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("[芯片检测] tbe 查询异常: %s", exc)
        return
    if result.returncode != 0:
        tail = result.stderr[-300:] if result.stderr else "(空)"
        logger.warning(
            "[芯片检测] tbe 失败(rc=%d): %s",
            result.returncode, tail,
        )
        return
    vals = result.stdout.strip().split("\n")
    if len(vals) < 6:
        return
    info["ub_size_kb"] = int(vals[0]) // 1024
    info["l1_size_kb"] = int(vals[1]) // 1024
    info["l0a_size_kb"] = int(vals[2]) // 1024
    info["l0b_size_kb"] = int(vals[3]) // 1024
    info["l0c_size_kb"] = int(vals[4]) // 1024
    info["ai_core_num"] = int(vals[5])
    info["detect_method"] = "tbe_device_query"


_SOC_MODEL_MAP = [
    (("950", "3510"), "Ascend 950", "dav-3510"),
    (("910b", "910c"), "Ascend 910B", "dav-c220"),
    (("310p",), "Ascend 310P", "dav-lite"),
    (("310",), "Ascend 310", "dav-mini"),
    (("910a", "910pro"), "Ascend 910A", "dav-c100"),
]


def _classify_chip_model(soc_name, info):
    """Map SoC name to chip_model and npu_arch."""
    soc_lower = soc_name.lower()
    for keywords, model, arch in _SOC_MODEL_MAP:
        if any(kw in soc_lower for kw in keywords):
            info["chip_model"] = model
            info["npu_arch"] = arch
            return


def detect_npu_device():
    """
    自动检测当前 NPU 设备信息。
    优先 tbe API 读真实参数，失败则返回 unknown。
    """
    info = {
        "chip_model": "unknown", "soc_version": "unknown",
        "npu_arch": "unknown", "ai_core_num": 0,
        "ub_size_kb": 0, "l1_size_kb": 0,
        "l0a_size_kb": 0, "l0b_size_kb": 0,
        "l0c_size_kb": 0, "l2_size_kb": 0,
        "cube_core_num": 0, "vector_core_num": 0,
        "device_id": 0, "detect_method": "none",
    }
    soc_name = _detect_soc_via_torch(info)
    if not soc_name:
        soc_name = _detect_soc_from_env()
        if soc_name:
            info["soc_version"] = soc_name
    if not soc_name:
        return info
    _query_tbe_params(soc_name, info)
    _classify_chip_model(soc_name, info)
    return info


def format_device_info_for_prompt(dev):
    """把芯片信息格式化为可注入 prompt 的文本。"""
    lines = [
        "当前 NPU 芯片信息（orchestrator 从设备读取）",
        "  芯片型号: " + dev["chip_model"],
        "  SoC 版本: " + dev["soc_version"],
        "  NPU 架构: " + dev["npu_arch"],
        "  AI Core: " + str(dev["ai_core_num"])
        + "(Cube " + str(dev.get("cube_core_num", "?"))
        + " + Vector " + str(dev.get("vector_core_num", "?"))
        + ")",
        "  UB: " + str(dev["ub_size_kb"]) + " KB",
        "  L1: " + str(dev["l1_size_kb"]) + " KB",
        "  L0A: " + str(dev.get("l0a_size_kb", "?")) + " KB",
        "  L0B: " + str(dev.get("l0b_size_kb", "?")) + " KB",
        "  L0C: " + str(dev.get("l0c_size_kb", "?")) + " KB",
        "  L2: " + str(dev.get("l2_size_kb", "?")) + " KB",
        "  Device ID: " + str(dev["device_id"]),
        "  检测方式: " + dev["detect_method"],
        "  ⚠️ tiling 参数必须按硬件参数计算，禁止写死",
    ]
    return "\n".join(lines) + "\n"


def load_config(config_path=None):
    """Load config from YAML."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "config.yaml"
        )
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fobj:
            return yaml.safe_load(fobj) or {}
    return {}


def append_work_record(work_dir, message):
    """Append timestamped record to WORK_RECORD.md."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(work_dir, "WORK_RECORD.md")
    with open(path, "a", encoding="utf-8") as fobj:
        fobj.write("\n## [" + ts + "] " + message + "\n")


def log_io(log_obj, stage, inputs, outputs):
    """Log stage inputs/outputs."""
    log_obj.info("[%s] 输入: %s", stage, ", ".join(inputs))
    log_obj.info(
        "[%s] 预期输出: %s", stage, ", ".join(outputs)
    )


def transition(state, new_stage, state_log, reason=""):
    """状态切换 + 写日志。"""
    old = state.current_stage
    state.current_stage = (
        "iter" + str(state.iteration) + "_" + new_stage
    )
    state.flush()
    reason_text = (" | " + reason) if reason else ""
    state_log.info(
        "%s → %s%s", old, state.current_stage, reason_text
    )


CANNBOT_CONSTRAINT = (
    "\n⚠️ 约束：所有临时文件必须放在 /tmp 下，"
    "禁止在工作目录创建临时文件。\n"
    "⚠️ 反作弊参考：knowledge/anti_cheat_reference.md"
    "（开发完成后按自检清单逐项确认）"
)


# ── Setup helpers ─────────────────────────────────────────


def _parse_and_setup(ctx):
    """Parse CLI args, load config, create work directory."""
    parser = argparse.ArgumentParser(
        description="PyPTO Pro Operator Workflow 控制平面"
    )
    parser.add_argument(
        "--task-dir", required=True,
        help="cannbench task 目录路径",
    )
    parser.add_argument("--op-name", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--init-impl", default=None)
    parser.add_argument("--optimize-hint", default=None)
    args = parser.parse_args()

    ctx.config = load_config(args.config)
    wf_cfg = ctx.config.get("workflow", {})
    ctx.max_iterations = (
        args.max_iter or wf_cfg.get("max_iterations", 20)
    )
    ctx.task_dir = os.path.abspath(args.task_dir)
    ctx.op_name = (
        args.op_name
        or os.path.basename(ctx.task_dir.rstrip("/"))
    )
    ctx.optimize_hint = args.optimize_hint or ""
    ctx.init_impl_mode = False
    _create_work_dir(ctx, args)
    _handle_init_impl(ctx, args)


def _create_work_dir(ctx, args):
    """Create and populate the work directory."""
    if args.work_dir:
        ctx.work_dir = os.path.abspath(args.work_dir)
    else:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        ctx.work_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "work", ctx.op_name + "_" + ts,
        )
    Path(ctx.work_dir).mkdir(parents=True, exist_ok=True)
    subdirs = [
        "impl", "build", "eval", "profile",
        "search", "develop", "log", "knowledge",
    ]
    for name in subdirs:
        Path(os.path.join(ctx.work_dir, name)).mkdir(
            parents=True, exist_ok=True
        )
    _create_symlinks(ctx)


def _create_symlinks(ctx):
    """Create task and example symlinks in work_dir."""
    task_link = os.path.join(ctx.work_dir, "task")
    if not os.path.exists(task_link):
        os.symlink(ctx.task_dir, task_link)
    example_link = os.path.join(ctx.work_dir, "example")
    if os.path.exists(example_link):
        return
    default_repo = (
        "/workspace/workflow_mutilagent_operator_develop"
        "/cann-bench"
    )
    repo = ctx.config.get("paths", {}).get(
        "cannbench_repo", default_repo,
    )
    src = os.path.join(repo, "examples", "pypto_pro_example")
    if os.path.isdir(src):
        os.symlink(src, example_link)


def _handle_init_impl(ctx, args):
    """Copy initial impl if --init-impl specified."""
    if not args.init_impl:
        return
    src = os.path.abspath(args.init_impl)
    if not os.path.isdir(src):
        logger.error("--init-impl 目录不存在: %s", src)
        sys.exit(1)
    if args.work_dir:
        logger.error("--init-impl 和 --work-dir 不能同时使用")
        sys.exit(1)
    dst = os.path.join(ctx.work_dir, "impl")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    ctx.init_impl_mode = True


def _setup_logging(ctx):
    """Initialize all loggers and state."""
    ctx.log = setup_logger(ctx.work_dir)
    ctx.state_log = setup_state_logger(ctx.work_dir)
    ctx.history_log = setup_history_logger(ctx.work_dir)
    ctx.node_logs = {
        ("N" + str(i)): setup_node_logger(
            ctx.work_dir, "N" + str(i)
        )
        for i in range(1, 6)
    }
    ctx.log.info("=" * 60)
    ctx.log.info("PyPTO Pro Operator Workflow 启动")
    ctx.log.info(
        "算子: %s | Task: %s", ctx.op_name, ctx.task_dir
    )
    ctx.log.info(
        "Work: %s | 最大迭代: %d",
        ctx.work_dir, ctx.max_iterations,
    )
    if ctx.init_impl_mode:
        ctx.log.info("⚡ 已有实现迭代模式")
    ctx.log.info("=" * 60)
    ctx.state = State.load_or_create(
        ctx.work_dir,
        op_name=ctx.op_name,
        max_iterations=ctx.max_iterations,
    )
    atexit.register(ctx.state.flush)
    signal.signal(
        signal.SIGTERM,
        lambda s, f: (ctx.state.flush(), sys.exit(1)),
    )
    signal.signal(
        signal.SIGINT,
        lambda s, f: (ctx.state.flush(), sys.exit(1)),
    )
    ctx.roles_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "roles"
    )


# ── Device detection and validation ──────────────────────


def _detect_device(ctx):
    """Detect NPU device and cache to device_info.json."""
    path = os.path.join(ctx.work_dir, "device_info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fobj:
            dev = json.load(fobj)
        ctx.log.info(
            "芯片信息已加载: %s (%s)",
            dev["chip_model"], dev["soc_version"],
        )
        ctx.device_info_prompt = format_device_info_for_prompt(
            dev
        )
        return
    hw = ctx.config.get("hardware", {})
    if hw.get("chip_model"):
        dev = _build_device_from_config(hw)
        ctx.log.info(
            "芯片信息来自 config.yaml: %s", dev["chip_model"]
        )
    else:
        dev = detect_npu_device()
        dev["device_id"] = hw.get("device_id", 0)
    _validate_device_info(ctx, dev)
    ctx.log.info(
        "芯片检测通过: %s (soc=%s, cores=%d, UB=%dKB)",
        dev["chip_model"], dev["soc_version"],
        dev["ai_core_num"], dev["ub_size_kb"],
    )
    atomic_write_json(path, dev)
    ctx.device_info_prompt = format_device_info_for_prompt(dev)


def _build_device_from_config(hw):
    """Build device_info dict from hardware config."""
    int_keys = [
        "ai_core_num", "ub_size_kb", "l1_size_kb",
        "l0a_size_kb", "l0b_size_kb", "l0c_size_kb",
        "l2_size_kb", "cube_core_num", "vector_core_num",
        "device_id",
    ]
    dev = {k: hw.get(k, 0) for k in int_keys}
    for k in ("chip_model", "soc_version", "npu_arch"):
        dev[k] = hw.get(k, "unknown")
    dev["detect_method"] = "config_manual"
    return dev


def _validate_device_info(ctx, dev):
    """Validate device info; exit if critical params missing."""
    missing = []
    if dev.get("soc_version", "unknown") == "unknown":
        missing.append("soc_version")
    no_ub = (
        dev.get("detect_method") == "none"
        or dev.get("ub_size_kb", 0) == 0
    )
    if no_ub:
        missing.append("UB_SIZE")
    if dev.get("l1_size_kb", 0) == 0:
        missing.append("L1_SIZE")
    if dev.get("ai_core_num", 0) == 0:
        missing.append("CORE_NUM")
    if dev.get("l0a_size_kb", 0) == 0:
        missing.append("L0A_SIZE")
    if not missing:
        return
    ctx.log.error("=" * 60)
    ctx.log.error("NPU 芯片参数检测失败，workflow 无法启动")
    ctx.log.error("缺失: %s", ", ".join(missing))
    ctx.log.error(
        "已获取: %s",
        json.dumps(dev, ensure_ascii=False, indent=2),
    )
    ctx.log.error("=" * 60)
    sys.exit(1)


# ── Stage 1 & 2 ──────────────────────────────────────────


def _run_stage1(ctx):
    """阶段1：需求分析（cannbot）。"""
    stage_match = (
        ctx.state.current_stage in ("N1_phase1", "stage1")
        or "stage1" in ctx.state.current_stage
    )
    if not stage_match:
        return
    stage = "阶段1-需求分析"
    wdir = ctx.work_dir
    role = os.path.join(
        ctx.roles_dir, "n1_stage1_requirements_analysis.md"
    )
    log_io(
        ctx.log, stage,
        [wdir + "/task/"], [wdir + "/ANALYSIS.md"],
    )
    append_work_record(wdir, stage)
    hint = _build_stage1_hint(ctx)
    prompt = (
        "工作目录：" + wdir + "\n算子：" + ctx.op_name + "\n"
        "task目录：" + wdir + "/task/\n"
        + ctx.device_info_prompt + "\n" + hint
        + "请分析task需求，输出写入 " + wdir + "/ANALYSIS.md"
        + CANNBOT_CONSTRAINT
    )
    ok = run_agent(
        "cannbot", role, wdir, prompt,
        node_log=ctx.node_logs["N1"],
    )
    if not ok:
        ctx.log.error("[%s] cannbot 失败", stage)
    transition(
        ctx.state, "stage2", ctx.state_log,
        reason="需求分析完成，开始编写首版",
    )


def _build_stage1_hint(ctx):
    """Build hint block for stage1 prompt."""
    if not ctx.optimize_hint:
        return ""
    if ctx.init_impl_mode:
        return (
            "\n⚡ 已有算子实现迭代模式：impl/ 下已有基线代码，"
            "请重点关注以下修改方向：\n"
            + ctx.optimize_hint + "\n"
        )
    return (
        "\n💡 融合方向提示（用户指定）：\n"
        + ctx.optimize_hint + "\n"
        "请在分析中重点关注该融合方向的可行性。\n"
    )


def _run_stage2(ctx):
    """阶段2：编写第一版算子（cannbot）。"""
    stage_match = (
        ctx.state.current_stage == "stage2"
        or "stage2" in ctx.state.current_stage
    )
    if not stage_match:
        return
    stage = "阶段2-编写第一版"
    wdir = ctx.work_dir
    role = os.path.join(
        ctx.roles_dir, "n1_stage2_first_impl.md"
    )
    design_dir = os.path.join(wdir, "develop", "iter0")
    Path(design_dir).mkdir(parents=True, exist_ok=True)
    log_io(
        ctx.log, stage,
        [wdir + "/task/", wdir + "/ANALYSIS.md"],
        [wdir + "/impl/" + ctx.op_name + "_impl.py"],
    )
    append_work_record(wdir, stage)
    prompt = _build_stage2_prompt(ctx, design_dir)
    ok = run_agent(
        "cannbot", role, wdir, prompt,
        node_log=ctx.node_logs["N1"],
    )
    if not ok:
        ctx.log.error("[%s] cannbot 失败", stage)
    transition(
        ctx.state, "ready_for_iter", ctx.state_log,
        reason="首版编写完成，进入迭代循环",
    )


def _build_stage2_prompt(ctx, design_dir):
    """Build prompt for stage2."""
    wdir = ctx.work_dir
    patterns = (
        format_proven_patterns_for_prompt(wdir)
        + format_regression_patterns_for_prompt(wdir)
        + format_pitfalls_for_prompt(wdir)
    )
    base = (
        "工作目录：" + wdir + "\n算子：" + ctx.op_name + "\n"
        "task目录：" + wdir + "/task/\n"
        "分析文档：" + wdir + "/ANALYSIS.md\n"
    )
    tail = (
        ctx.device_info_prompt + "\n" + patterns + "\n"
        "设计思路输出到 " + design_dir
        + "/design_rationale.md\n"
        "自测报告输出到 " + design_dir
        + "/self_test_report.md" + CANNBOT_CONSTRAINT
    )
    hint = _stage2_hint(ctx)
    if ctx.init_impl_mode:
        return (
            base + "现有代码：" + wdir + "/impl/\n"
            "参考示例：" + wdir + "/example/\n"
            + hint + tail
        )
    return (
        base + "参考示例：" + wdir + "/example/\n" + hint
        + "请编写算子，输出到 " + wdir + "/impl/"
        + ctx.op_name + "_impl.py\n" + tail
    )


def _stage2_hint(ctx):
    """Build hint block for stage2 prompt."""
    if ctx.init_impl_mode:
        hint = (
            "\n⚡ 已有基线代码，请完成首次修改。\n"
        )
        if ctx.optimize_hint:
            hint += (
                "修改方向：" + ctx.optimize_hint + "\n"
                "请按此方向修改，不要从零重写。\n"
            )
        return hint
    if ctx.optimize_hint:
        return (
            "\n💡 融合方向提示：\n" + ctx.optimize_hint + "\n"
            "请在实现中按此融合方向设计数据流。\n"
        )
    return ""


# ── Iteration loop stages ────────────────────────────────


def _get_resume_stage(current_stage):
    """Extract stage number from current_stage string."""
    if "stage" in current_stage:
        try:
            return int(current_stage.split("stage")[-1])
        except ValueError:
            pass
    return 0


def _create_iter_dirs(work_dir, iteration):
    """Create and return per-iteration directory dict."""
    name = "iter" + str(iteration)
    dirs = {
        "build": os.path.join(work_dir, "build", name),
        "eval": os.path.join(work_dir, "eval", name),
        "profile": os.path.join(work_dir, "profile", name),
        "search": os.path.join(work_dir, "search", name),
        "develop": os.path.join(work_dir, "develop", name),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


def _find_self_test(ctx):
    """Find most recent self_test_report.md."""
    for i in range(ctx.state.iteration, -1, -1):
        cand = os.path.join(
            ctx.work_dir, "develop",
            "iter" + str(i), "self_test_report.md",
        )
        if os.path.exists(cand):
            return cand
    return None


def _run_stage4(ctx, iter_dirs):
    """阶段4：编译部署（kerminal）。Returns True if success."""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段4-编译"
    )
    role = os.path.join(ctx.roles_dir, "n2_stage4_build.md")
    self_test = _find_self_test(ctx)
    hint = ("\n自测报告：" + self_test) if self_test else ""
    inputs = [ctx.work_dir + "/impl/"]
    if self_test:
        inputs.append(self_test)
    outputs = [iter_dirs["build"] + "/build.log"]
    log_io(ctx.log, stage, inputs, outputs)
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage4", ctx.state_log,
        reason="开始编译部署",
    )
    prompt = (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "impl目录：" + ctx.work_dir + "/impl/\n"
        + ctx.device_info_prompt + "\n"
        "请编译部署，结果写入 "
        + iter_dirs["build"] + "/build.log" + hint
    )
    ok = run_agent(
        "kerminal", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N2"],
    )
    log_path = iter_dirs["build"] + "/build.log"
    build_ok = check_build_success(log_path)
    if not ok and build_ok:
        ctx.log.warning(
            "[%s] kerminal 异常但 build.log 显示 SUCCESS",
            stage,
        )
    if build_ok:
        return True
    ctx.log.warning("[%s] 编译失败 → 阶段3", stage)
    ctx.state_log.info("[判断] 编译失败")
    append_work_record(ctx.work_dir, stage + " FAIL → 阶段3")
    append_ledger_only(
        ctx.work_dir, ctx.state.iteration, "build_fail"
    )
    run_tech_lead(ctx, iter_dirs, "build_fail")
    self_goto_stage3(ctx, iter_dirs, "build_fail")
    return False


def _run_stage5(ctx, iter_dirs):
    """阶段5：精度评测。Returns precision dict or None."""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段5-精度评测"
    )
    role = os.path.join(
        ctx.roles_dir, "n2_stage5_precision_eval.md"
    )
    log_io(
        ctx.log, stage,
        ["task_dir=" + ctx.task_dir],
        [iter_dirs["eval"] + "/precision_result.json"],
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage5", ctx.state_log,
        reason="编译成功，开始精度评测",
    )
    prompt = (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "task_dir：" + ctx.task_dir + "\n"
        + ctx.device_info_prompt + "\n"
        "结果写入 " + iter_dirs["eval"]
        + "/precision_result.json"
    )
    run_agent(
        "kerminal", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N2"],
    )
    prec_path = iter_dirs["eval"] + "/precision_result.json"
    precision = parse_precision_result(prec_path)
    _link_precision_report(iter_dirs, precision)
    if precision.get("precision_overall", False):
        return precision
    _handle_precision_fail(ctx, iter_dirs, precision, stage)
    return None


def _handle_precision_fail(ctx, iter_dirs, precision, stage):
    """Handle precision failure: log, tech_lead, stage3."""
    passed = precision.get("passed_cases", "?")
    total = precision.get("total_cases", "?")
    ctx.log.warning(
        "[%s] 精度不通过(%s/%s) → 阶段3",
        stage, passed, total,
    )
    ctx.state_log.info(
        "[判断] 精度失败: passed=%s/%s", passed, total
    )
    append_work_record(ctx.work_dir, stage + " FAIL → 阶段3")
    append_ledger_only(
        ctx.work_dir, ctx.state.iteration, "precision_fail"
    )
    run_tech_lead(ctx, iter_dirs, "precision_fail")
    self_goto_stage3(ctx, iter_dirs, "precision_fail")


def _link_precision_report(iter_dirs, precision):
    """Create symlink for precision report if available."""
    src = precision.get("source_report", "")
    if not src or not os.path.exists(src):
        return
    link_dir = os.path.join(
        iter_dirs["eval"], "precision_reports"
    )
    Path(link_dir).mkdir(parents=True, exist_ok=True)
    link_path = os.path.join(link_dir, os.path.basename(src))
    if not os.path.exists(link_path):
        os.symlink(src, link_path)


# ── Stage 6: Performance evaluation ──────────────────────


def _run_stage6_eval(ctx, iter_dirs):
    """阶段6：执行 cann-bench 性能评测。Returns perf dict."""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段6-性能评测"
    )
    log_io(
        ctx.log, stage,
        ["task_dir=" + ctx.task_dir],
        [iter_dirs["eval"] + "/perf_result.json"],
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage6", ctx.state_log,
        reason="精度通过，开始性能评测",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_cfg = ctx.config.get("paths", {}).get(
        "cannbench_repo", "../cann-bench"
    )
    cannbench_src = os.path.abspath(
        os.path.join(script_dir, repo_cfg, "src")
    )
    if not os.path.isdir(cannbench_src):
        cannbench_src = os.path.abspath(
            os.path.join(script_dir, repo_cfg)
        )
    ctx.log.info("[%s] 执行 cann-bench 性能评测", stage)
    ctx.node_logs["N2"].info(
        "性能评测开始，task=%s", ctx.task_dir
    )
    success, output, report = _run_perf_with_retry(
        ctx, cannbench_src, stage
    )
    csv_dir = _find_csv_dir(ctx, cannbench_src)
    perf = _parse_perf_report(report, success, output, csv_dir)
    atomic_write_json(
        iter_dirs["eval"] + "/perf_result.json", perf
    )
    ctx.log.info(
        "[%s] perf: overall=%s, avg=%s",
        stage, perf.get("overall_score"),
        perf.get("avg_speedup"),
    )
    _copy_prof_data(ctx, iter_dirs, cannbench_src, stage)
    _link_perf_reports(iter_dirs, perf)
    return perf


def _run_perf_with_retry(ctx, cannbench_src, stage):
    """Run perf eval with up to 2 retries."""
    dev_id = ctx.config.get("hardware", {}).get("device_id", 0)
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        success, output, report = run_perf_eval(
            ctx.task_dir, cannbench_src,
            device_id=dev_id,
            log_callback=lambda m: ctx.node_logs["N2"].info(m),
        )
        ctx.node_logs["N2"].info(
            "性能评测第%d/%d次, success=%s, report=%s",
            attempt, max_retries, success, report,
        )
        ctx.log.info(
            "[%s] 第%d次: success=%s, report=%s",
            stage, attempt, success,
            "有" if report else "无",
        )
        if success or report:
            return success, output, report
        if attempt < max_retries:
            ctx.log.warning(
                "[%s] 崩溃（第%d次），10秒后重试",
                stage, attempt,
            )
            time.sleep(10)
            continue
        ctx.log.error(
            "[%s] 连续%d次崩溃，停止", stage, max_retries,
        )
        append_work_record(
            ctx.work_dir,
            stage + " cann-bench 连续崩溃",
        )
        ctx.state.flush()
        raise RuntimeError(
            "cann-bench 连续" + str(max_retries) + "次崩溃"
        )
    return False, "", None


def _find_csv_dir(ctx, cannbench_src):
    """Locate profiling CSV directory."""
    parts = Path(ctx.task_dir).parts
    level = next(
        (p for p in parts if p.startswith("level")), ""
    )
    if not level:
        return ""
    parent = os.path.dirname(os.path.dirname(cannbench_src))
    csv_dir = os.path.join(
        parent, "reports", "prof_data", level, ctx.op_name,
    )
    if os.path.isdir(csv_dir):
        return csv_dir
    return os.path.join(
        cannbench_src, "reports", "prof_data",
        level, ctx.op_name,
    )


def _parse_perf_report(report_path, success, output, csv_dir):
    """Parse cann-bench report into perf dict."""
    if report_path:
        return parse_cannbench_report_to_perf_result(
            report_path, csv_dir
        )
    if not success:
        tail = output[-200:] if output else "无输出"
        logger.error("评测崩溃且无新报告")
        return {
            "perf_pass": False, "overall_score": 0.0,
            "avg_speedup": 0.0,
            "score_error_code": "eval_crash",
            "score_error": "评测崩溃(rc=" + tail + ")",
            "_error": "cann-bench 评测崩溃",
        }
    logger.error("cann-bench 未产出报告")
    return {
        "perf_pass": False, "overall_score": 0.0,
        "avg_speedup": 0.0,
        "_error": "cann-bench 未产出报告",
    }


def _copy_prof_data(ctx, iter_dirs, cannbench_src, stage):
    """Copy profiling data to iter eval directory."""
    parts = Path(ctx.task_dir).parts
    level = next(
        (p for p in parts if p.startswith("level")), ""
    )
    if not level:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_cfg = ctx.config.get("paths", {}).get(
        "cannbench_repo", "../cann-bench"
    )
    repo_abs = os.path.abspath(
        os.path.join(script_dir, repo_cfg)
    )
    prof_src = os.path.join(
        repo_abs, "reports", "prof_data",
        level, ctx.op_name,
    )
    if not os.path.isdir(prof_src):
        prof_src = os.path.join(
            cannbench_src, "reports", "prof_data",
            level, ctx.op_name,
        )
    prof_dst = os.path.join(iter_dirs["eval"], "prof_data")
    if not os.path.isdir(prof_src):
        ctx.log.warning(
            "[%s] prof_data 不存在: %s", stage, prof_src
        )
        return
    if os.path.exists(prof_dst):
        shutil.rmtree(prof_dst)
    shutil.copytree(prof_src, prof_dst)
    ctx.log.info(
        "[%s] prof_data 已复制 (%d cases)",
        stage, len(os.listdir(prof_dst)),
    )


def _link_perf_reports(iter_dirs, perf):
    """Create symlinks for perf report files."""
    src_json = perf.get("source_json", "")
    if not src_json or not os.path.exists(src_json):
        return
    link_dir = os.path.join(iter_dirs["eval"], "perf_reports")
    Path(link_dir).mkdir(parents=True, exist_ok=True)
    link_path = os.path.join(
        link_dir, os.path.basename(src_json)
    )
    if not os.path.exists(link_path):
        os.symlink(src_json, link_path)
    for ext in (".md", ".html"):
        related = src_json.replace(".json", ext)
        if not os.path.exists(related):
            continue
        rlink = os.path.join(
            link_dir, os.path.basename(related)
        )
        if not os.path.exists(rlink):
            os.symlink(related, rlink)


# ── Stage 6 routing ──────────────────────────────────────


def _update_history_after_stage6(
    ctx, iter_dirs, perf, precision, resume
):
    """Update state history and history.json after stage6."""
    prec_ok = (
        precision.get("precision_overall", False)
        if resume <= 5
        else True
    )
    ctx.state.update_last_eval({
        "precision_overall": prec_ok,
        "perf_pass": perf.get("perf_pass", False),
        "perf_speedup": perf.get("overall_score", 0.0),
    })
    prev_speedup = 0.0
    if len(ctx.state.history) >= 2:
        prev_speedup = ctx.state.history[-2].get(
            "speedup", 0.0
        )
    append_round(
        ctx.work_dir, ctx.state.iteration,
        perf, prev_avg_speedup=prev_speedup,
    )
    hist = load_history(ctx.work_dir)
    ctx.history_log.info(
        "ITER%d [stage6]\n%s",
        ctx.state.iteration,
        json.dumps(hist, ensure_ascii=False, indent=2),
    )


def _route_after_stage6(ctx, iter_dirs, perf, perf_diff):
    """Route: anticheat / perf_pass / optimize.

    Returns 'continue', 'break', or 'normal'.
    """
    score_err = (
        perf.get("score_error_code")
        or perf.get("score_error", "")
    )
    avg_sp = perf.get("avg_speedup", 0.0)
    is_anticheat = bool(score_err) or avg_sp == 0.0
    if is_anticheat:
        _handle_anticheat(ctx, iter_dirs, perf, perf_diff)
        return "continue"
    if perf.get("perf_pass", False):
        return _handle_perf_pass(ctx, iter_dirs, perf_diff)
    ctx.state_log.info(
        "[判断] 性能不达标: avg=%s → stage7-9-3", avg_sp
    )
    return "normal"


def _handle_anticheat(ctx, iter_dirs, perf, perf_diff):
    """Handle anticheat / zero-score route."""
    score_err = (
        perf.get("score_error_code")
        or perf.get("score_error", "")
    )
    avg_sp = perf.get("avg_speedup", 0.0)
    reason = (
        score_err if score_err
        else "avg_speedup=0（所有case零）"
    )
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段6-性能评测"
    )
    ctx.state_log.info(
        "[判断] 反作弊触发: %s, avg=%s", reason, avg_sp
    )
    csv_analysis = analyze_kernel_csv_for_anticheat(perf)
    ctx.log.warning(
        "[%s] 反作弊/零分: %s → 阶段3", stage, reason
    )
    ctx.log.info("[%s] kernel_csv:\n%s", stage, csv_analysis)
    append_work_record(
        ctx.work_dir, stage + " 反作弊 → 阶段3"
    )
    anticheat_ctx = (
        "score_error_code: " + str(score_err) + "\n"
        "性能结果：" + iter_dirs["eval"]
        + "/perf_result.json\n"
        "profiler：" + iter_dirs["eval"] + "/prof_data/\n"
        "\n=== kernel_csv 分析 ===\n" + csv_analysis + "\n"
    )
    directive = _build_anticheat_directive(
        reason, csv_analysis
    )
    run_tech_lead(
        ctx, iter_dirs,
        "score_zero: " + reason + "\n" + directive,
        perf_diff,
    )
    self_goto_stage3(
        ctx, iter_dirs, "score_zero", anticheat_ctx
    )


def _build_anticheat_directive(reason, csv_analysis):
    """Build anticheat directive text for tech_lead."""
    return (
        "\n⛔ 反作弊触发！分析为什么触发。⛔\n"
        "score_error_code: " + reason + "\n"
        "\n=== csv 分析 ===\n" + csv_analysis + "\n"
        "\n步骤：\n"
        "1. 读 knowledge/anti_cheat_reference.md\n"
        "2. 确定匹配的反作弊编号\n"
        "3. 按「怎么避免」给出修改方案\n"
        "4. ⛔ 禁止缓存优化/sleep 等绕道方案\n"
    )


def _handle_perf_pass(ctx, iter_dirs, perf_diff):
    """Handle perf-pass route. Returns 'break' or 'continue'."""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段6-性能评测"
    )
    ctx.state_log.info("[判断] 性能达标 → 融合验证")
    ctx.log.info("[%s] ⚡ 性能达标 → stage9 融合验证", stage)
    append_work_record(
        ctx.work_dir, stage + " 性能达标 → stage9"
    )
    directive = (
        "\n⚡ 性能已达标，验证融合是否真融合。\n"
        "1. 读最新 design_rationale.md\n"
        "2. 读 impl/ kernel 代码\n"
        "3. 检查 kernel_details.csv\n"
        "4. 填写 exit_decision: true/false\n"
    )
    run_tech_lead(
        ctx, iter_dirs,
        "perf_pass_fusion_verify\n" + directive,
        perf_diff,
    )
    hist = load_history(ctx.work_dir)
    exit_dec = hist.get("exit_decision")
    if isinstance(exit_dec, str):
        exit_dec = exit_dec.lower() == "true"
    if exit_dec is True:
        ctx.state_log.info("[判断] 真融合+达标 → 退出")
        ctx.log.info("[%s] ✅ 真融合+达标 → 退出", stage)
        ctx.state.stopped_by = "target_met_fusion_verified"
        ctx.state.flush()
        return "break"
    ctx.state_log.info("[判断] 假融合 → 继续迭代")
    ctx.log.info("[%s] ❌ 假融合 → 继续迭代", stage)
    self_goto_stage3(ctx, iter_dirs, "fusion_improve")
    return "continue"


# ── Stages 7, 8, 10 ──────────────────────────────────────


def _run_stage7(ctx, iter_dirs, perf_diff_text):
    """阶段7：Profiling 分析（kerminal）。"""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段7-profiling"
    )
    role = os.path.join(
        ctx.roles_dir, "n2_stage7_kerminal_profile.md"
    )
    log_io(
        ctx.log, stage,
        [iter_dirs["eval"] + "/perf_result.json"],
        [iter_dirs["profile"] + "/bottleneck_analysis.md"],
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage7", ctx.state_log,
        reason="性能不达标，开始profiling",
    )
    prev_design = _find_prev_design(ctx)
    patterns = (
        format_proven_patterns_for_prompt(ctx.work_dir)
        + format_regression_patterns_for_prompt(ctx.work_dir)
        + format_pitfalls_for_prompt(ctx.work_dir)
    )
    prompt = (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "性能结果：" + iter_dirs["eval"]
        + "/perf_result.json\n"
        "报告目录：" + iter_dirs["eval"] + "/perf_reports/\n"
        "上轮设计：" + prev_design + "\n"
        + ctx.device_info_prompt + "\n"
        + perf_diff_text + "\n" + patterns + "\n"
        "输出 " + iter_dirs["profile"]
        + "/bottleneck_analysis.md"
    )
    run_agent(
        "kerminal", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N2"],
    )


def _find_prev_design(ctx):
    """Find most recent design_rationale.md."""
    for i in range(ctx.state.iteration - 1, -1, -1):
        cand = os.path.join(
            ctx.work_dir, "develop",
            "iter" + str(i), "design_rationale.md",
        )
        if os.path.exists(cand):
            return cand
    return os.path.join(
        ctx.work_dir, "develop", "iter0",
        "design_rationale.md",
    )


def _run_stage8(ctx, iter_dirs, perf_diff_text):
    """阶段8：搜索优化方案（hermes）。"""
    stage = (
        "ITER" + str(ctx.state.iteration) + "-阶段8-搜索"
    )
    role = os.path.join(ctx.roles_dir, "n3_stage8_search.md")
    log_io(
        ctx.log, stage,
        [iter_dirs["profile"] + "/bottleneck_analysis.md"],
        [iter_dirs["search"] + "/FIX_DIRECTIVE.md"],
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage8", ctx.state_log,
        reason="profiling完成，开始搜索",
    )
    patterns = (
        format_proven_patterns_for_prompt(ctx.work_dir)
        + format_regression_patterns_for_prompt(ctx.work_dir)
    )
    impl_path = (
        ctx.work_dir + "/impl/" + ctx.op_name + "_impl.py"
    )
    prompt = (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "瓶颈：" + iter_dirs["profile"]
        + "/bottleneck_analysis.md\n"
        "代码：" + impl_path + "\n"
        + ctx.device_info_prompt + "\n"
        + perf_diff_text + "\n" + patterns + "\n"
        "输出：" + iter_dirs["search"]
        + "/SEARCH_REPORT.md 和 "
        + iter_dirs["search"] + "/FIX_DIRECTIVE.md"
    )
    run_agent(
        "hermes", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N3"],
    )


def _run_stage10(ctx, iter_dirs):
    """阶段10：最终报告（kerminal）。"""
    stage = "阶段10-最终报告"
    role = os.path.join(
        ctx.roles_dir, "n5_stage10_kerminal_report.md"
    )
    log_io(
        ctx.log, stage,
        [iter_dirs["eval"] + "/precision_result.json",
         iter_dirs["eval"] + "/perf_result.json"],
        [ctx.work_dir + "/FINAL_REPORT.md"],
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage10", ctx.state_log,
        reason="生成最终报告",
    )
    q_hint = _check_unhandled_questions(ctx, stage)
    prompt = (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "精度：" + iter_dirs["eval"]
        + "/precision_result.json\n"
        "性能：" + iter_dirs["eval"]
        + "/perf_result.json\n"
        "记录：" + ctx.work_dir + "/WORK_RECORD.md\n"
        "历史：" + ctx.work_dir + "/knowledge/history.json\n"
        + q_hint + ctx.device_info_prompt + "\n"
        "输出 " + ctx.work_dir + "/FINAL_REPORT.md"
    )
    run_agent(
        "kerminal", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N5"],
    )


def _check_unhandled_questions(ctx, stage):
    """Check for unhandled question.md files."""
    unhandled = []
    pattern = os.path.join(
        ctx.work_dir, "develop", "iter*", "question.md"
    )
    for qpath in glob.glob(pattern):
        with open(qpath, "r", encoding="utf-8") as fobj:
            content = fobj.read()
        if "## tech_lead 裁定" not in content:
            unhandled.append(qpath)
    if not unhandled:
        return ""
    ctx.log.warning(
        "[%s] %d 个未裁定的 question.md",
        stage, len(unhandled),
    )
    return (
        "未裁定的下级反馈：" + ", ".join(unhandled) + "\n"
    )


# ── Main iteration loop + entry ──────────────────────────


def _run_iterations(ctx):
    """Main iteration loop. Returns last iter_dirs."""
    iter_dirs = None
    while ctx.state.iteration < ctx.state.max_iterations:
        resume = _get_resume_stage(ctx.state.current_stage)
        is_resume = 3 < resume <= 9
        if is_resume:
            ctx.log.info(
                "%s ITER %d/%d (从stage%d恢复) %s",
                "-" * 40, ctx.state.iteration,
                ctx.state.max_iterations, resume, "-" * 40,
            )
        else:
            ctx.state.iteration += 1
            ctx.state.flush()
            ctx.log.info(
                "%s ITER %d/%d %s",
                "-" * 40, ctx.state.iteration,
                ctx.state.max_iterations, "-" * 40,
            )
            resume = 4
        iter_dirs = _create_iter_dirs(
            ctx.work_dir, ctx.state.iteration
        )
        action = _run_single_iteration(
            ctx, iter_dirs, resume
        )
        if action == "break":
            break
    return iter_dirs


def _run_single_iteration(ctx, iter_dirs, resume):
    """Run one iteration. Returns 'break', 'continue', or None."""
    precision = None
    if resume <= 4:
        if not _run_stage4(ctx, iter_dirs):
            return "continue"
    if resume <= 5:
        precision = _run_stage5(ctx, iter_dirs)
        if precision is None:
            return "continue"
    if resume <= 6:
        perf = _run_stage6_eval(ctx, iter_dirs)
        _update_history_after_stage6(
            ctx, iter_dirs, perf, precision or {}, resume
        )
        perf_diff = _compute_perf_diff(
            ctx.work_dir, ctx.state.iteration
        )
        _log_perf_diff(ctx, perf_diff)
        route = _route_after_stage6(
            ctx, iter_dirs, perf, perf_diff
        )
        if route in ("continue", "break"):
            return route
        perf_diff_text = (
            perf_diff.get("diff_text", "")
            + perf_diff.get("regression_text", "")
        )
    else:
        perf_diff = _compute_perf_diff(
            ctx.work_dir, ctx.state.iteration
        )
        perf_diff_text = (
            perf_diff.get("diff_text", "")
            + perf_diff.get("regression_text", "")
        )
    if resume <= 7:
        _run_stage7(ctx, iter_dirs, perf_diff_text)
    if resume <= 8:
        _run_stage8(ctx, iter_dirs, perf_diff_text)
    if resume <= 9:
        run_tech_lead(
            ctx, iter_dirs, "perf_optimize", perf_diff
        )
    self_goto_stage3(ctx, iter_dirs, "perf_optimize")
    return None


def _log_perf_diff(ctx, perf_diff):
    """Log perf diff summary to state log."""
    if perf_diff.get("has_improvement"):
        ctx.state_log.info(
            "[判断] 性能提升: %s->%s (+%s%%)",
            perf_diff["prev_avg_speedup"],
            perf_diff["curr_avg_speedup"],
            perf_diff["delta_pct"],
        )
        return
    if perf_diff.get("has_regression"):
        ctx.state_log.info(
            "[判断] 性能退步: %s->%s (%s%%)",
            perf_diff["prev_avg_speedup"],
            perf_diff["curr_avg_speedup"],
            perf_diff["delta_pct"],
        )
        return
    delta = perf_diff.get("delta_pct", 0)
    if delta != 0:
        ctx.state_log.info(
            "[判断] 性能变化不显著: delta=%s%%", delta
        )
    else:
        ctx.state_log.info("[判断] 性能diff无法计算")


def main():
    """Entry point for PyPTO Pro Operator Workflow.

    sys.exit is used here for CLI error handling as this
    is the main entry point.
    """
    ctx = _WfCtx()
    _parse_and_setup(ctx)
    _setup_logging(ctx)
    _detect_device(ctx)
    _run_stage1(ctx)
    _run_stage2(ctx)
    iter_dirs = _run_iterations(ctx)
    max_hit = (
        ctx.state.iteration >= ctx.state.max_iterations
        and not ctx.state.stopped_by
    )
    if max_hit:
        ctx.state.stopped_by = "max_iterations"
        ctx.log.warning(
            "达到最大迭代次数 %d", ctx.state.max_iterations
        )
    if iter_dirs is None:
        iter_dirs = _create_iter_dirs(
            ctx.work_dir, ctx.state.iteration
        )
    _run_stage10(ctx, iter_dirs)
    ctx.log.info("=" * 60)
    ctx.log.info("完成！stopped_by=%s", ctx.state.stopped_by)
    ctx.log.info("报告：%s/FINAL_REPORT.md", ctx.work_dir)
    ctx.log.info("=" * 60)
    ctx.state.flush()


# ── Perf diff computation ────────────────────────────────


def _compute_perf_diff(work_dir, iteration):
    """Compute performance diff between current and prev iter."""
    result = {
        "has_improvement": False, "has_regression": False,
        "diff_text": "", "regression_text": "",
    }
    curr_path = os.path.join(
        work_dir, "eval", "iter" + str(iteration),
        "perf_result.json",
    )
    if not os.path.exists(curr_path):
        return result
    try:
        with open(curr_path, "r", encoding="utf-8") as fobj:
            curr_perf = json.load(fobj)
    except (OSError, json.JSONDecodeError):
        return result
    curr_avg = curr_perf.get("avg_speedup", 0)
    prev_avg, prev_iter, prev_perf = _find_prev_perf(
        work_dir, iteration
    )
    if prev_avg <= 0 or curr_avg <= 0:
        return result
    delta_pct = round(
        (curr_avg - prev_avg) / prev_avg * 100, 1
    )
    result.update({
        "prev_iter": prev_iter,
        "curr_iter": iteration,
        "prev_avg_speedup": prev_avg,
        "curr_avg_speedup": curr_avg,
        "delta_pct": delta_pct,
        "has_improvement": delta_pct >= 5,
        "has_regression": delta_pct <= -5,
    })
    _add_design_paths(result, work_dir, iteration, prev_iter)
    result["case_diffs"] = _compute_case_diffs(
        curr_perf, prev_perf
    )
    _format_perf_diff_text(
        result, delta_pct, prev_avg, curr_avg
    )
    return result


def _find_prev_perf(work_dir, iteration):
    """Find previous valid perf. Returns (avg, iter, dict)."""
    for i in range(iteration - 1, 0, -1):
        path = os.path.join(
            work_dir, "eval", "iter" + str(i),
            "perf_result.json",
        )
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fobj:
                prev = json.load(fobj)
            avg = prev.get("avg_speedup", 0)
            if avg > 0:
                return avg, i, prev
        except (OSError, json.JSONDecodeError):
            continue
    return 0, 0, {}


def _add_design_paths(result, work_dir, curr_iter, prev_iter):
    """Add design_rationale paths to result dict."""
    for tag, start in [("curr_design_path", curr_iter),
                       ("prev_design_path", prev_iter)]:
        for i in range(start, -1, -1):
            p = os.path.join(
                work_dir, "develop", "iter" + str(i),
                "design_rationale.md",
            )
            if os.path.exists(p):
                result[tag] = p
                break


def _compute_case_diffs(curr_perf, prev_perf):
    """Compute per-case speedup diffs."""
    prev_map = {}
    for c in prev_perf.get("worst_6_cases", []):
        cid = c.get("case_id", c.get("id", ""))
        if cid:
            prev_map[str(cid)] = c.get("speedup", 0)
    diffs = []
    for c in curr_perf.get("worst_6_cases", []):
        cid = str(c.get("case_id", c.get("id", "")))
        curr_sp = c.get("speedup", 0)
        prev_sp = prev_map.get(cid, 0)
        if prev_sp <= 0:
            continue
        cdelta = round(
            (curr_sp - prev_sp) / prev_sp * 100, 1
        )
        sign = "+" if cdelta > 0 else ""
        diffs.append({
            "case_id": cid,
            "prev_speedup": prev_sp,
            "curr_speedup": curr_sp,
            "delta": sign + str(cdelta) + "%",
        })
    diffs.sort(
        key=lambda x: float(
            x["delta"].replace("%", "").replace("+", "")
        ),
        reverse=True,
    )
    return diffs


def _format_perf_diff_text(result, delta, prev, curr):
    """Populate diff_text / regression_text in result."""
    if result["has_improvement"]:
        result["diff_text"] = _fmt_improvement(
            result, delta, prev, curr
        )
    elif result["has_regression"]:
        result["regression_text"] = _fmt_regression(
            result, delta, prev, curr
        )


def _append_round_and_design(lines, result):
    """Append round info and design paths common to both improvement/regression."""
    lines.append(
        "  轮次: iter" + str(result["prev_iter"])
        + " -> iter" + str(result["curr_iter"])
    )
    if result.get("prev_design_path"):
        lines.append(
            "  上轮设计: " + result["prev_design_path"]
        )
    if result.get("curr_design_path"):
        lines.append(
            "  本轮设计: " + result["curr_design_path"]
        )


def _fmt_improvement(result, delta, prev, curr):
    """Format improvement text."""
    lines = [
        "\n性能提升（程序自动计算）",
        "  avg_speedup: " + str(prev) + " -> "
        + str(curr) + " (+" + str(delta) + "%)",
    ]
    _append_round_and_design(lines, result)
    for cd in result.get("case_diffs", [])[:6]:
        lines.append(
            "    case_" + cd["case_id"] + ": "
            + str(cd["prev_speedup"]) + " -> "
            + str(cd["curr_speedup"])
            + " (" + cd["delta"] + ")"
        )
    lines.append(
        "\n⚡ 性能提升！请总结改动为什么有效，"
        "填写 proven_pattern。"
    )
    return "\n".join(lines)


def _fmt_regression(result, delta, prev, curr):
    """Format regression text."""
    lines = [
        "\n⚠️ 性能退步（程序自动计算）",
        "  avg_speedup: " + str(prev) + " -> "
        + str(curr) + " (" + str(delta) + "%)",
    ]
    _append_round_and_design(lines, result)
    sorted_diffs = sorted(
        result.get("case_diffs", []),
        key=lambda x: float(
            x["delta"].replace("%", "").replace("+", "")
        ),
    )
    for cd in sorted_diffs[:6]:
        lines.append(
            "    case_" + cd["case_id"] + ": "
            + str(cd["prev_speedup"]) + " -> "
            + str(cd["curr_speedup"])
            + " (" + cd["delta"] + ")"
        )
    lines.append(
        "\n🚨 性能退步！分析原因，"
        "填写 regression_pattern。"
    )
    return "\n".join(lines)


# ── Tech Lead (stage 9) ──────────────────────────────────


def run_tech_lead(ctx, iter_dirs, fail_reason, perf_diff=None):
    """Call tech_lead to update history.json."""
    iteration = ctx.state.iteration
    stage = (
        "ITER" + str(iteration)
        + "-阶段9-tech_lead(" + fail_reason + ")"
    )
    role = os.path.join(
        ctx.roles_dir, "n4_stage9_tech_lead_guide.md"
    )
    history_path = os.path.join(
        ctx.work_dir, "knowledge", "history.json"
    )
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage9", ctx.state_log,
        reason="tech_lead(" + fail_reason + ")",
    )
    if perf_diff is None:
        perf_diff = _compute_perf_diff(
            ctx.work_dir, iteration
        )
    prompt = _build_tech_lead_prompt(
        ctx, iter_dirs, fail_reason,
        perf_diff, history_path,
    )
    ctx.log.info("[%s] 启动 tech_lead", stage)
    save_pre_tech_lead_snapshot(ctx.work_dir)
    run_agent(
        "kerminal", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N4"],
    )
    restore_program_fields(ctx.work_dir)
    hist = load_history(ctx.work_dir)
    question_path, _ = _find_pending_question(ctx)
    _record_knowledge(ctx, hist, perf_diff, question_path, stage)
    save_history(ctx.work_dir, hist)
    _log_tech_lead_update(ctx, hist, fail_reason)


def _build_tech_lead_prompt(
    ctx, iter_dirs, fail_reason, perf_diff, history_path
):
    """Assemble tech_lead prompt from available data."""
    data_lines = _collect_tech_lead_data(iter_dirs)
    design_hint = _collect_all_designs(ctx)
    _, q_hint = _find_pending_question(ctx)
    diff_hint = (
        perf_diff.get("diff_text", "")
        + perf_diff.get("regression_text", "")
    )
    patterns = (
        format_proven_patterns_for_prompt(ctx.work_dir)
        + format_regression_patterns_for_prompt(ctx.work_dir)
        + format_pitfalls_for_prompt(ctx.work_dir)
    )
    anti_cheat_ref = os.path.join(
        os.path.dirname(ctx.roles_dir),
        "knowledge", "anti_cheat_reference.md",
    )
    return (
        "工作目录：" + ctx.work_dir + "\n"
        "算子：" + ctx.op_name + "\n"
        "task目录：" + ctx.work_dir + "/task/\n"
        "需求分析：" + ctx.work_dir + "/ANALYSIS.md\n"
        "反作弊参考：" + anti_cheat_ref + "\n"
        "历史经验：" + history_path + "\n"
        + ctx.device_info_prompt + "\n"
        "本轮失败原因：" + fail_reason + "\n"
        + patterns + diff_hint + q_hint + "\n"
        + "\n".join(data_lines) + "\n"
        + design_hint
        + "请分析本轮结果，更新 " + history_path
    )


def _collect_tech_lead_data(iter_dirs):
    """Collect available data file paths for tech_lead."""
    candidates = [
        ("性能结果", iter_dirs["eval"] + "/perf_result.json"),
        ("profiler", iter_dirs["eval"] + "/prof_data/"),
        ("性能报告", iter_dirs["eval"] + "/perf_reports/"),
        ("瓶颈分析",
         iter_dirs["profile"] + "/bottleneck_analysis.md"),
        ("修改指令",
         iter_dirs["search"] + "/FIX_DIRECTIVE.md"),
        ("编译日志", iter_dirs["build"] + "/build.log"),
        ("精度结果",
         iter_dirs["eval"] + "/precision_result.json"),
        ("精度报告",
         iter_dirs["eval"] + "/precision_reports/"),
        ("自测报告",
         iter_dirs["develop"] + "/self_test_report.md"),
    ]
    return [
        name + "：" + path
        for name, path in candidates
        if os.path.exists(path)
    ]


def _collect_all_designs(ctx):
    """Collect all design_rationale.md paths."""
    designs = []
    for i in range(0, ctx.state.iteration + 1):
        cand = os.path.join(
            ctx.work_dir, "develop",
            "iter" + str(i), "design_rationale.md",
        )
        if os.path.exists(cand):
            designs.append("iter" + str(i) + ": " + cand)
    if not designs:
        return ""
    return "历轮设计：\n" + "\n".join(designs) + "\n"


def _find_pending_question(ctx):
    """Find unresolved question.md from previous iter."""
    for i in range(ctx.state.iteration - 1, -1, -1):
        cand = os.path.join(
            ctx.work_dir, "develop",
            "iter" + str(i), "question.md",
        )
        if not os.path.exists(cand):
            continue
        with open(cand, "r", encoding="utf-8") as fobj:
            content = fobj.read()
        if "## tech_lead 裁定" in content:
            continue
        hint = (
            "\n⚖️ 下级反馈待裁定：" + cand + "\n"
            "请按「第零步D」裁定。\n"
        )
        return cand, hint
    return None, ""


# ── Knowledge recording ──────────────────────────────────


def _record_knowledge(ctx, hist, perf_diff, qpath, stage):
    """Record proven/regression patterns and pitfalls."""
    _record_proven(ctx, hist, perf_diff, stage)
    _record_regression(ctx, hist, perf_diff, stage)
    _record_pitfalls(ctx, hist, qpath, stage)


def _record_proven(ctx, hist, perf_diff, stage):
    """Record proven patterns from performance improvement."""
    if not perf_diff.get("has_improvement"):
        hist.pop("_pending_proven_pattern", None)
        return
    pending = hist.pop("_pending_proven_pattern", None)
    base = {
        "speedup_before": perf_diff["prev_avg_speedup"],
        "speedup_after": perf_diff["curr_avg_speedup"],
        "delta_pct": perf_diff["delta_pct"],
    }
    if pending and isinstance(pending, dict):
        pending.update(base)
        append_proven_pattern(
            ctx.work_dir, ctx.state.iteration, pending
        )
        ctx.log.info(
            "[%s] 成功经验: delta=%s%%",
            stage, perf_diff["delta_pct"],
        )
        return
    base.update({
        "what_changed": "(tech_lead 未填写)",
        "why_it_worked": "(tech_lead 未填写)",
        "fusion_related": False,
    })
    append_proven_pattern(
        ctx.work_dir, ctx.state.iteration, base
    )
    ctx.log.warning(
        "[%s] 提升 %s%% 但未填 proven_pattern",
        stage, perf_diff["delta_pct"],
    )


def _record_regression(ctx, hist, perf_diff, stage):
    """Record regression patterns from performance drop."""
    if not perf_diff.get("has_regression"):
        hist.pop("_pending_regression_pattern", None)
        return
    pending = hist.pop("_pending_regression_pattern", None)
    base = {
        "speedup_before": perf_diff["prev_avg_speedup"],
        "speedup_after": perf_diff["curr_avg_speedup"],
        "delta_pct": perf_diff["delta_pct"],
    }
    if pending and isinstance(pending, dict):
        pending.update(base)
        append_regression_pattern(
            ctx.work_dir, ctx.state.iteration, pending
        )
        ctx.log.info(
            "[%s] 退步教训: delta=%s%%",
            stage, perf_diff["delta_pct"],
        )
        return
    base.update({
        "what_changed": "(tech_lead 未填写)",
        "why_it_failed": "(tech_lead 未填写)",
        "fusion_related": False,
    })
    append_regression_pattern(
        ctx.work_dir, ctx.state.iteration, base
    )
    ctx.log.warning(
        "[%s] 退步 %s%% 但未填 regression_pattern",
        stage, perf_diff["delta_pct"],
    )


def _record_pitfalls(ctx, hist, question_path, stage):
    """Record pitfall from question.md resolution."""
    if not question_path:
        hist.pop("_pending_pitfall", None)
        return
    pending = hist.pop("_pending_pitfall", None)
    if not (pending and isinstance(pending, dict)):
        ctx.log.warning(
            "[%s] question.md 存在但无 pitfall 裁定", stage
        )
        return
    append_pitfall(ctx.work_dir, ctx.state.iteration, pending)
    verdict = pending.get("verdict", "confirmed")
    _write_verdict_to_question(ctx, question_path, pending)
    append_work_record(
        ctx.work_dir, stage + " 裁定: " + verdict
    )
    ctx.log.info(
        "[%s] 错题本: verdict=%s", stage, verdict
    )


def _write_verdict_to_question(ctx, qpath, pending):
    """Append tech_lead verdict to question.md."""
    verdict = pending.get("verdict", "confirmed")
    try:
        with open(qpath, "a", encoding="utf-8") as fobj:
            fobj.write(
                "\n\n## tech_lead 裁定"
                "（iter" + str(ctx.state.iteration) + "）\n"
            )
            is_confirmed = verdict == "confirmed"
            label = "确认误判" if is_confirmed else "驳回"
            fobj.write("- 结论: " + label + "\n")
            cause = pending.get("root_cause", "?")
            fobj.write("- 原因: " + cause + "\n")
            approach = pending.get("correct_approach", "?")
            fobj.write("- 正确做法: " + approach + "\n")
    except OSError as exc:
        ctx.log.warning(
            "写入裁定失败: %s", exc
        )


def _log_tech_lead_update(ctx, hist, fail_reason):
    """Log tech_lead's history update."""
    for name in ("pypto-pro-history", "ascendc-history"):
        hl = logging.getLogger(name)
        if hl.handlers:
            hl.info(
                "ITER%d [stage9 tech_lead(%s)]\n%s",
                ctx.state.iteration, fail_reason,
                json.dumps(hist, ensure_ascii=False, indent=2),
            )
            break


# ── Stage 3: Fix & Optimize (cannbot) ────────────────────


def self_goto_stage3(ctx, iter_dirs, reason, extra_ctx=""):
    """阶段3：修改优化（cannbot）。"""
    stage = (
        "ITER" + str(ctx.state.iteration)
        + "-阶段3-修改(" + reason + ")"
    )
    role = os.path.join(
        ctx.roles_dir, "n1_stage3_fix_and_optimize.md"
    )
    _backup_impl(ctx, stage)
    append_work_record(ctx.work_dir, stage)
    transition(
        ctx.state, "stage3", ctx.state_log,
        reason="回退修改(" + reason + ")",
    )
    prompt = _build_stage3_prompt(
        ctx, iter_dirs, reason, extra_ctx
    )
    inputs, outputs = _stage3_io_paths(
        ctx, iter_dirs, reason
    )
    log_io(ctx.log, stage, inputs, outputs)
    prompt += CANNBOT_CONSTRAINT
    ok = run_agent(
        "cannbot", role, ctx.work_dir, prompt,
        node_log=ctx.node_logs["N1"],
    )
    if not ok:
        ctx.log.error("[%s] cannbot 失败", stage)


def _backup_impl(ctx, stage):
    """Backup current impl to operator_iter/."""
    backup_dir = os.path.join(
        ctx.work_dir, "operator_iter",
        "iter" + str(ctx.state.iteration),
    )
    impl_dir = os.path.join(ctx.work_dir, "impl")
    if not os.path.isdir(impl_dir):
        return
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(impl_dir, backup_dir)
    ctx.log.info("[%s] impl 备份到 %s", stage, backup_dir)


def _build_stage3_prompt(ctx, iter_dirs, reason, extra_ctx):
    """Build full prompt for stage3 cannbot."""
    wdir = ctx.work_dir
    op = ctx.op_name
    base = (
        "工作目录：" + wdir + "\n"
        "算子：" + op + "\n"
        "当前代码：" + wdir + "/impl/" + op + "_impl.py\n"
        "task目录：" + wdir + "/task/\n"
        "需求分析：" + wdir + "/ANALYSIS.md\n"
        + ctx.device_info_prompt + "\n"
        "回退原因：" + reason + "\n"
    )
    pitfalls = format_pitfalls_for_prompt(wdir)
    if pitfalls:
        base += "\n" + pitfalls + "\n"
    base += (
        "\n如发现 tech_lead 建议不可行，可写反馈到 "
        + iter_dirs["develop"] + "/question.md\n"
    )
    fix_plan = get_latest_fix_plan(wdir)
    if fix_plan:
        base += "\n" + fix_plan + "\n\n"
    return base + _stage3_reason_block(
        ctx, iter_dirs, reason, extra_ctx
    )


def _stage3_reason_block(ctx, iter_dirs, reason, extra_ctx):
    """Build reason-specific block for stage3 prompt."""
    dev = iter_dirs["develop"]
    history_ctx = format_for_prompt(ctx.work_dir)
    test_hint = "自测报告输出到 " + dev + "/self_test_report.md\n"
    design_hint = (
        "设计思路输出到 " + dev + "/design_rationale.md\n"
    )
    exp_block = "\n=== 历史经验 ===\n" + history_ctx + "\n"
    if reason == "build_fail":
        return (
            "编译日志：" + iter_dirs["build"] + "/build.log\n"
            "请修复编译错误。\n" + test_hint + exp_block
        )
    if reason == "precision_fail":
        return (
            "精度结果：" + iter_dirs["eval"]
            + "/precision_result.json\n"
            "精度报告：" + iter_dirs["eval"]
            + "/precision_reports/\n"
            "请修复精度问题。\n" + test_hint + exp_block
        )
    if reason == "score_zero":
        return (
            "⚠️ cann-bench 性能全零，反作弊触发。\n"
            + extra_ctx + "\n请排查并修复。\n"
            + design_hint + test_hint + exp_block
        )
    if reason == "fusion_improve":
        return (
            "⚡ 性能达标但融合不充分。\n"
            "性能结果：" + iter_dirs["eval"]
            + "/perf_result.json\n"
            + design_hint + test_hint + exp_block + "\n"
            "**重点**：改进融合方案。\n"
        )
    # perf_optimize
    return (
        "性能结果：" + iter_dirs["eval"]
        + "/perf_result.json\n"
        "性能报告：" + iter_dirs["eval"]
        + "/perf_reports/\n"
        "瓶颈：" + iter_dirs["profile"]
        + "/bottleneck_analysis.md\n"
        "搜索：" + iter_dirs["search"]
        + "/SEARCH_REPORT.md\n"
        "指令：" + iter_dirs["search"]
        + "/FIX_DIRECTIVE.md\n"
        + design_hint + test_hint + exp_block + "\n"
        "请按指令优化代码。\n"
    )


def _stage3_io_paths(ctx, iter_dirs, reason):
    """Return (inputs, outputs) for stage3."""
    wdir = ctx.work_dir
    op = ctx.op_name
    outputs = [wdir + "/impl/" + op + "_impl.py"]
    impl_path = wdir + "/impl/" + op + "_impl.py"
    if reason == "build_fail":
        return (
            [impl_path, iter_dirs["build"] + "/build.log"],
            outputs,
        )
    if reason == "precision_fail":
        return (
            [impl_path,
             iter_dirs["eval"] + "/precision_result.json"],
            outputs,
        )
    if reason in ("score_zero", "fusion_improve"):
        return (
            [impl_path,
             iter_dirs["eval"] + "/perf_result.json"],
            outputs,
        )
    return (
        [impl_path,
         iter_dirs["search"] + "/FIX_DIRECTIVE.md",
         iter_dirs["profile"] + "/bottleneck_analysis.md"],
        outputs,
    )


# sys.exit is used in main() as this is the CLI entry point.
if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
