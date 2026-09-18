# PyPTO Pro Workflow 环境迁移指南

> 本文档指导如何在一台新机器上从零配置 PyPTO Pro Workflow 的完整运行环境。
>
> **目标**：只改配置文件中的路径和 API Key，不改代码，让 workflow 框架可以跑在任何有昇腾 NPU 的系统上。

---

## 前置条件

| 依赖 | 要求 | 说明 |
|------|------|------|
| 昇腾 NPU | 在线可用 | `npu-smi info` 能看到设备 |
| CANN Toolkit | ≥ 9.1.0 已安装 | 包含 PyPTO 0.2.1 |
| Python | ≥ 3.8 | 建议 3.10+ |
| Node.js | ≥ 18 | CANNBot 依赖 |
| Git | 已安装 | 拉代码用 |

---

## 第 1 步：CANN + PyPTO 环境

> PyPTO 内置在 CANN Toolkit 中，不需要单独安装。只需确保 CANN 环境变量正确配置。

### 1.1 配置环境变量

在 `~/.bashrc` 末尾添加（**路径按你的实际 CANN 安装目录调整**）：

```bash
# CANN 环境变量（修改 CANN_PATH 为你的实际路径）
export CANN_PATH=/home/developer/Ascend/cann-9.1.0

export LD_LIBRARY_PATH=$CANN_PATH/lib64:$CANN_PATH/x86_64-linux/lib64:$CANN_PATH/lib64/plugin/opskernel:$CANN_PATH/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=$CANN_PATH/python/site-packages:$PYTHONPATH
export PATH=$CANN_PATH/bin:$PATH

source ~/.bashrc
```

### 1.2 验证

```bash
# 验证 NPU
npu-smi info

# 验证 PyTorch + NPU
python3 -c "import torch, torch_npu; print(f'torch={torch.__version__}, npu={torch.npu.is_available()}')"

# 验证 PyPTO
python3 -c "import pypto; print('PyPTO 可用')"
```

### 1.3 PyPTO Pro（可选，实验性）

PyPTO Pro（SPMD 模式）尚未在已发布的 CANN 中提供，需从源码编译。详见 [docs/pypto_build_guide.md](pypto_build_guide.md)。

**当前无 PyPTO Pro 时**：框架会自动退回 `pypto.frontend` 栈，功能正常但小 shape case 性能受限（~35μs AICPU 调度开销）。

---

## 第 2 步：Kerminal 安装

> 最简单的一步，一条命令搞定。

### 2.1 安装

```bash
curl -fsSL https://kerminal.cn/install.sh | bash
```

安装后位于 `~/.local/bin/kerminal`。

### 2.2 配置

```bash
mkdir -p ~/.kerminal
cat > ~/.kerminal/config.toml << 'EOF'
model = "kernelcat1.0"
model_provider = "autokernel"
thinking_enabled = true
thinking_budget_tokens = 10000
show_thinking = true
show_raw_agent_reasoning = true

[features]
skills = true

[model_providers.autokernel]
name = "autokernel"
wire_api = "messages"
experimental_bearer_token = "<你的 Kerminal API Key>"
EOF
```

> **API Key 获取**：联系 Autokernel 团队获取，或使用已有的 Key。

### 2.3 验证

```bash
kerminal --version
# 应输出: kerminal x.x.x

# 测试 CLI 是否可调用（非交互模式）
echo "你好" | kerminal --print -p "回复OK"
```

---

## 第 3 步：Hermes 安装

### 3.1 安装

```bash
# 方式 A：官方一键安装脚本（推荐）
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# 方式 B：从源码安装（如果有 hermes-agent-main 目录）
cd hermes-agent-main && pip install -e .
```

安装后会创建 `~/.hermes/` 目录和 venv。

如果 `/usr/local/bin/hermes` 不存在或报 ModuleNotFoundError，做软链接：

```bash
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes
```

### 3.2 配置

需要配置两个文件：

**~/.hermes/config.yaml**（模型配置）：

```bash
cat > ~/.hermes/config.yaml << 'EOF'
model:
  default: glm/glm-5.3-flash
EOF
```

**~/.hermes/.env**（API Key + 搜索引擎）：

```bash
cat > ~/.hermes/.env << 'EOF'
GLM_API_KEY=<你的智谱 API Key>
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
TAVILY_API_KEY=<你的 Tavily API Key，可选，没有则用免费 ddgs>
EOF
```

### 3.3 能否直接复制配置到另一台机器？

**可以复制的**（配置文件，跨机器通用）：
- `~/.hermes/config.yaml` — 模型配置
- `~/.hermes/.env` — API Key（只要 Key 没过期就能用）

**不能复制的**（需要重新安装）：
- `~/.hermes/venvs/` — Python venv，包含编译的 .so 文件，跨系统不兼容
- `/usr/local/bin/hermes` — 软链接，目标路径可能不同

**迁移建议**：在新机器上重新运行安装脚本，然后把 `config.yaml` 和 `.env` 复制过去即可。

### 3.4 验证

```bash
# 检查版本
hermes --version
# 应输出: Hermes Agent vX.X.X

# 检查状态
hermes status

# 测试 CLI 是否可调用（非交互模式）
hermes -z "回复OK" --cli
```

---

## 第 4 步：CANNBot 安装

### 4.1 前置：Node.js

```bash
# 检查 Node.js（需要 ≥ 18）
node --version

# 如果没有 Node.js，用 nvm 安装
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 18
```

### 4.2 安装 CANNBot

```bash
npm install -g cannbot @cannbot-ai/install-helper

# 验证
cannbot --version
# 应输出: 1.1.2+
```

### 4.3 配置

```bash
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/opencode.jsonc << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM (ZhipuAI)",
      "options": {
        "baseURL": "https://open.bigmodel.cn/api/paas/v4",
        "apiKey": "<你的智谱 API Key>"
      },
      "models": {
        "glm-5.3-flash": { "name": "GLM-5.3-Flash" }
      }
    }
  },
  "model": "glm/glm-5.3-flash"
}
EOF
```

### 4.4 启动方式

```bash
# 交互界面（TUI）
cannbot

# 带初始消息的交互
cannbot "帮我分析这个算子"

# 非交互模式（CLI 调用，workflow 用的就是这个）
cannbot run "你好，回复OK"

# Web 界面
cannbot web
```

### 4.5 验证

```bash
# 版本
cannbot --version

# 测试 CLI 是否可调用（非交互，发一条消息）
cannbot run "回复OK"
```

---

## 第 5 步：Workflow 配置

### 5.1 克隆项目

```bash
git clone https://gitcode.com/AI4SE/pypto-pro-workflow.git
cd pypto-pro-workflow
```

### 5.2 修改 config.yaml

**只需要改这一个文件**，把路径改成你的实际路径：

```yaml
agents:
  cannbot:
    cli: "<你的 cannbot 路径>"     # which cannbot 查看
  kerminal:
    cli: "<你的 kerminal 路径>"    # which kerminal 查看
  hermes:
    cli: "<你的 hermes 路径>"      # which hermes 查看

paths:
  cannbench_repo: "<你的 cann-bench 路径>"   # cann-bench 仓库的本地路径
```

查找各工具路径：

```bash
echo "cannbot: $(which cannbot)"
echo "kerminal: $(which kerminal)"
echo "hermes: $(which hermes)"
```

### 5.3 修改 agent_runner.py 中的 CANN 路径（如果 CANN 安装位置不同）

`lib/agent_runner.py` 中硬编码了 CANN 9.1.0 的路径。如果你的 CANN 安装在不同位置，需要修改以下几行：

```python
# lib/agent_runner.py 第 31-51 行
cann_pypath = "/usr/local/Ascend/cann/python/site-packages"          # 改成你的
cann91_pypath = "/home/developer/Ascend/cann-9.1.0/python/site-packages"  # 改成你的
cann91_tbe = "/home/developer/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe"  # 改成你的
cann_bin = "/home/developer/Ascend/cann-9.1.0/bin"                   # 改成你的
cann_ccec = "/home/developer/Ascend/cann-9.1.0/tools/ccec_compiler/bin"  # 改成你的
```

> **TODO（后续优化）**：将这些路径提取到 `config.yaml` 中，消除代码中的硬编码。

### 5.4 安装 cann-bench

```bash
git clone https://gitcode.com/cann/cann-bench.git
# 确保 config.yaml 中的 cannbench_repo 路径指向这个目录
```

---

## 第 6 步：验证全流程

### 6.1 逐个验证 Agent

```bash
# 1. Kerminal
kerminal --version

# 2. Hermes
hermes --version

# 3. CANNBot
cannbot --version

# 4. NPU
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
```

### 6.2 运行 Workflow

```bash
cd pypto-pro-workflow
python orchestrator.py --task-dir <cann-bench>/tasks/level1/exp --max-iter 3
```

如果一切正常，会自动走完 阶段1→阶段2→迭代循环（阶段4→5→6→7→8→9→3）。

---

## API Key 清单

迁移时需要配置以下 API Key（**不要硬编码在代码中**）：

| API Key | 用途 | 配置文件位置 | 获取方式 |
|---------|------|-------------|---------|
| Kerminal (Autokernel) | Kerminal Agent 模型调用 | `~/.kerminal/config.toml` → `experimental_bearer_token` | 联系 Autokernel |
| 智谱 GLM | CANNBot + Hermes 模型调用 | CANNBot: `~/.config/opencode/opencode.jsonc` → `apiKey`<br>Hermes: `~/.hermes/.env` → `GLM_API_KEY` | https://open.bigmodel.cn |
| Tavily（可选） | Hermes 搜索引擎 | `~/.hermes/.env` → `TAVILY_API_KEY` | https://tavily.com |
| CANNBench（可选） | Kerminal 的 benchsite MCP | `~/.kerminal/config.toml` → `BENCHSITE_API_TOKEN` | cann-bench 平台 |

---

## 迁移检查清单

在新机器上逐项确认：

- [ ] CANN Toolkit 已安装，`npu-smi info` 正常
- [ ] `~/.bashrc` 中 CANN 环境变量已配置，`source ~/.bashrc` 后 `python3 -c "import pypto"` 正常
- [ ] Kerminal 已安装，`kerminal --version` 正常
- [ ] `~/.kerminal/config.toml` 已配置（含 API Key）
- [ ] Hermes 已安装，`hermes --version` 正常
- [ ] `~/.hermes/config.yaml` + `~/.hermes/.env` 已配置
- [ ] Node.js ≥ 18 已安装，`node --version` 正常
- [ ] CANNBot 已安装，`cannbot --version` 正常
- [ ] `~/.config/opencode/opencode.jsonc` 已配置（含 API Key）
- [ ] cann-bench 已克隆到本地
- [ ] `pypto-pro-workflow/config.yaml` 中的 3 个 CLI 路径 + 1 个 cann-bench 路径已修改
- [ ] `lib/agent_runner.py` 中的 CANN 路径已修改（如果安装位置不同）
- [ ] `python orchestrator.py --task-dir ... --max-iter 1` 能跑通第一轮

---

## 常见问题

### Q: `hermes` 命令报 ModuleNotFoundError

```bash
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes
```

### Q: `cannbot run` 报连接错误

检查 `~/.config/opencode/opencode.jsonc` 中的 `apiKey` 和 `baseURL` 是否正确。

### Q: orchestrator 报 "CANN 环境检测失败"

确保 `~/.bashrc` 中的 CANN 环境变量已 source，且 `python3 -c "import torch_npu"` 不报错。

### Q: 某个 Agent 超时无响应

检查对应的 API Key 是否过期，网络是否能访问模型 API（`curl https://open.bigmodel.cn/api/paas/v4/models`）。

### Q: 想换芯片型号（如从 950 换到 910B）

不需要改代码——orchestrator 启动时会自动检测当前芯片型号并注入到所有 stage。只需确保 CANN 环境变量指向正确版本的 Toolkit。
