# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile


_HELPER_CONTINUE = object()

LOGGER = logging.getLogger(__name__)


def _get_info_from_anaconda_info(info, split=":"):
    info = info.strip("\n").replace(" ", "")
    info_dict = {}
    latest_key = ""
    for line in info.splitlines():
        if split in line:
            pair = line.split(split)
            info_dict[pair[0]] = pair[1]
            latest_key = pair[0]
        else:
            if not isinstance(info_dict.get(latest_key), list):
                info_dict[latest_key] = [info_dict.get(latest_key)]
            info_dict.get(latest_key).append(line)
    return info_dict


def find_anaconda():
    # try find in default path
    path = Path.home() / "anaconda3"
    if path.exists():
        return path
    # try conda in cmd
    try:
        conda = shutil.which("conda")
        if conda is None:
            raise FileNotFoundError("conda executable not found")
        info = subprocess.check_output([conda, "info"]).decode("utf-8")
        info_dict = _get_info_from_anaconda_info(info)
        return info_dict.get("activeenvlocation")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("find anaconda failed") from exc


def find_cuda():
    """Finds the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        if sys.platform == "win32":
            cuda_homes = glob.glob(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*"
            )
            if len(cuda_homes) == 0:
                cuda_home = ""
            else:
                cuda_home = cuda_homes[0]
        else:
            cuda_home = "/usr/local/cuda"
        if not os.path.exists(cuda_home):
            # Guess #3
            try:
                which = "where" if sys.platform == "win32" else "which"
                nvcc = subprocess.check_output([which, "nvcc"]).decode().rstrip("\r\n")
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    if cuda_home is None:
        raise RuntimeError(
            "No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home)
        )
    return cuda_home


def _compile_cuda_device_query(cuda_home):
    source = """
            #include <cuda_runtime.h>
            #include <iostream>
            int main(){
                int nDevices;
                cudaGetDeviceCount(&nDevices);
                for (int i = 0; i < nDevices; i++) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, i);
                    std::cout << prop.major << "." << prop.minor << std::endl;
                }
                return 0;
            }
            """
    with tempfile.NamedTemporaryFile("w", suffix=".cc") as file_handle:
        source_path = Path(file_handle.name)
        file_handle.write(source)
        file_handle.flush()
        compiler = shutil.which("g++")
        if compiler is None:
            return None
        command = [
            compiler,
            file_handle.name,
            "-o",
            source_path.stem,
            f"-I{cuda_home / 'include'}",
            f"-L{cuda_home / 'lib64'}",
            "-lcudart",
        ]
        LOGGER.info("%s", command)
        subprocess.check_output(command, cwd=source_path.parent)
        executable = source_path.parent / source_path.stem
        arches = (
            subprocess.check_output([str(executable)])
            .decode()
            .rstrip("\r\n")
            .split("\n")
        )
        return arches[0] if arches else None


def _query_cuda_arch(cuda_home):
    device_query_path = cuda_home / "extras/demo_suite/deviceQuery"
    if not device_query_path.exists():
        return _compile_cuda_device_query(cuda_home)
    output = subprocess.check_output([str(device_query_path)]).decode()
    capability = next(line for line in output.splitlines() if "CUDA Capability" in line)
    return capability.split()[-1]


def _find_supported_cuda_arch(arch):
    major, minor = (int(part) for part in arch.split("."))
    arch_int = major * 10 + minor
    while arch_int > 10:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            return None
        try:
            subprocess.check_output(
                [nvcc, f"-arch=sm_{arch_int}"], stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as error:
            output = error.output.decode()
            if "No input files specified" in output:
                return f"sm_{arch_int}"
            if "is not defined for option 'gpu-architecture'" in output:
                arch_int -= 1
                continue
            raise RuntimeError("unknown error") from error
    return None


def find_cuda_device_arch():
    if sys.platform == "win32":
        return None
    cuda_home = find_cuda()
    if cuda_home is None:
        return None
    try:
        arch = _query_cuda_arch(Path(cuda_home))
        return _find_supported_cuda_arch(arch) if arch is not None else None

    except Exception:
        return None


def _module_get_gpu_memory_usage_157_part_1(_state):
    if sys.platform == "win32":
        return None
    _state["cuda_home"] = find_cuda()
    if _state["cuda_home"] is None:
        return None
    _state["cuda_home"] = Path(_state["cuda_home"])
    _state[
        "source"
    ] = """
    #include <cuda_runtime.h>
    #include <iostream>
    int main(){
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        size_t free_m, total_m;
        // output json format.
        std::cout << "[";
        for (int i = 0; i < nDevices; i++) {
            cudaSetDevice(i);
            cudaMemGetInfo(&free_m, &total_m);
            std::cout << "[" << free_m << "," << total_m << "]";
            if (i != nDevices - 1)
                std::cout << "," << std::endl;
        }
        std::cout << "]" << std::endl;
        return 0;
    }
    """
    return _HELPER_CONTINUE


def _module_get_gpu_memory_usage_157_part_2(_state):
    with tempfile.NamedTemporaryFile("w", suffix=".cc") as _state["f"]:
        _state["f_path"] = Path(_state["f"].name)
        _state["f"].write(_state["source"])
        _state["f"].flush()
        try:
            _state["compiler"] = shutil.which("g++")
            if _state["compiler"] is None:
                return None
            _state["cmd"] = [
                _state["compiler"],
                _state["f"].name,
                "-o",
                _state["f_path"].stem,
                "-std=c++11",
                f"-I{_state['cuda_home'] / 'include'}",
                f"-L{_state['cuda_home'] / 'lib64'}",
                "-lcudart",
            ]
            LOGGER.info("%s", _state["cmd"])
            subprocess.check_output(_state["cmd"], cwd=_state["f_path"].parent)
            _state["executable"] = _state["f_path"].parent / _state["f_path"].stem
            _state["usages"] = subprocess.check_output(
                [str(_state["executable"])]
            ).decode()
            _state["usages"] = json.loads(_state["usages"])
            return _state["usages"]
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            return None
    return _HELPER_CONTINUE


def get_gpu_memory_usage():
    _state = locals()
    _helper_result = _module_get_gpu_memory_usage_157_part_1(_state)
    if _helper_result is not _HELPER_CONTINUE:
        return _helper_result
    _helper_result = _module_get_gpu_memory_usage_157_part_2(_state)
    if _helper_result is not _HELPER_CONTINUE:
        return _helper_result
    return None


if __name__ == "__main__":
    LOGGER.info("%s", find_cuda_device_arch())
