# Issue #001: NPU 芯片参数检测失败（Segmentation fault / ImportError）

## 问题现象

orchestrator 启动时，芯片自动检测阶段报错退出，表现为两种形式：

**形式 A：主进程直接 import torch 时段错误**
```
Fatal Python error: Segmentation fault
Current thread 0x00007f0d9d81f7c0 (most recent call first):
  File "/opt/buildtools/python-3.12.9/lib/python3.12/ctypes/__init__.py", line 157 in <module>
  File "/opt/buildtools/python-3.12.9/lib/python3.12/site-packages/torch/__init__.py", line 14 in <module>
  File "orchestrator.py", line 65 in detect_npu_device
```

**形式 B：子进程 import torch 时 ImportError**
```
ImportError: /opt/buildtools/python-3.12.9/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so: undefined symbol: _PyErr_SetLocaleString
```

两种形式交替出现，取决于是否带 `LD_PRELOAD`。

## 问题定位

### 根因

`/opt/buildtools/python-3.12.9/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so` 文件损坏。

这个环境中 `/usr/local/bin/python3` → `/opt/buildtools/python-3.12.9/bin/python3`，所有 `python3` 调用都走这个坏的 `_ctypes.so`。该文件缺少 `_PyErr_SetLocaleString` 符号（Python 3.12.3 标准库需要，但 3.12.9 构建时未包含）。

### 证据链

1. `ls` 发现同目录下有 `.broken`、`.broken2`、`.broken_current` 等备份——说明之前有人尝试修过但没成功
2. 系统原装 `_ctypes.so` 在 `/usr/lib/python3.12/lib-dynload/` 下完好
3. 替换后立即修复

### 触发条件

- 带 `LD_PRELOAD=/lib/x86_64-linux-gnu/libgomp.so.1` 时 → SIGSEGV（段错误）
- 不带 `LD_PRELOAD` 时 → ImportError（undefined symbol）
- CANN 驱动默认设置了 `LD_PRELOAD`，所以从 `source ~/.bashrc` 的终端跑通常是形式 A

## 解决方案

### 直接修复（已执行）

```bash
cp /usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so \
   /opt/buildtools/python-3.12.9/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so
```

### 代码层面防御（已实现）

1. **芯片检测用 subprocess 隔离**（不在主进程 import torch），防止段错误崩主进程
2. **subprocess 用 `bash -c "source ~/.bashrc; python3 脚本"` 执行**，自带完整环境
3. **重试 3 次间隔 5 秒**，应对 NPU 设备临时未释放
4. **失败时打印 stderr + 排查步骤**，不再吃掉错误信息
5. **`faulthandler.enable()`** 在主进程入口开启，段错误时打印 Python 调用栈

### 验证

```bash
python3 -c "from _ctypes import Union; import torch, torch_npu; print(torch_npu.npu.get_device_properties(0).name)"
# 应输出: Ascend950PR_9579
```

## 影响范围

- orchestrator.py 的 `detect_npu_device()` 函数
- 所有依赖 `python3` + `import torch` 的 subprocess 调用
