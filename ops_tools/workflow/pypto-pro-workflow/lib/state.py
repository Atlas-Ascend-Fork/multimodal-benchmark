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

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .handoff import atomic_write_json


@dataclass
class State:
    schema_version: int = 1
    op_name: str = ""
    work_dir: str = ""
    current_stage: str = "N1_phase1"
    iteration: int = 0
    max_iterations: int = 20
    last_eval: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    stopped_by: Optional[str] = None

    _path: str = field(default="", repr=False)

    @classmethod
    def load_or_create(
        cls, work_dir: str, op_name: str = "",
        max_iterations: int = 20,
    ) -> "State":
        state_path = os.path.join(work_dir, ".state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            st = cls(
                schema_version=data.get("schema_version", 1),
                op_name=data.get("op_name", op_name),
                work_dir=data.get("work_dir", work_dir),
                current_stage=data.get(
                    "current_stage", "N1_phase1",
                ),
                iteration=data.get("iteration", 0),
                max_iterations=data.get(
                    "max_iterations", max_iterations,
                ),
                last_eval=data.get("last_eval", {}),
                history=data.get("history", []),
                stopped_by=data.get("stopped_by"),
            )
            st._path = state_path
            return st
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        st = cls(
            op_name=op_name, work_dir=work_dir,
            max_iterations=max_iterations,
        )
        st._path = state_path
        st.flush()
        return st

    def flush(self):
        data = {
            "schema_version": self.schema_version,
            "op_name": self.op_name,
            "work_dir": self.work_dir,
            "current_stage": self.current_stage,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "last_eval": self.last_eval,
            "history": self.history,
            "stopped_by": self.stopped_by,
        }
        atomic_write_json(self._path, data)

    def update_last_eval(self, eval_result: Dict[str, Any]):
        prec = eval_result.get("precision_overall", False)
        perf = eval_result.get("perf_pass", False)
        spd = eval_result.get("perf_speedup", 0.0)
        self.last_eval = {
            "precision_pass": prec,
            "perf_pass": perf,
            "perf_speedup": spd,
        }
        self.history.append({
            "iteration": self.iteration,
            "precision_pass": prec,
            "perf_pass": perf,
            "speedup": spd,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
        self.flush()

    def get_recent_history(self, n: int = 5) -> str:
        recent = self.history[-n:]
        lines = []
        for h in recent:
            prec_str = "PASS" if h.get("precision_pass") else "FAIL"
            spd_str = h.get("speedup", "N/A")
            lines.append(
                f"- iter{h['iteration']}: "
                f"prec={prec_str}, speedup={spd_str}"
            )
        return "\n".join(lines) if lines else "(no history)"
