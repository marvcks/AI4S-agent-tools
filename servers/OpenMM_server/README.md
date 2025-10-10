# OpenMM MCP 服务器

本项目提供了一个基于 CalculationMCPServer 的服务器，用于使用 OpenMM 进行分子动力学模拟。

## 安装

安装所有必需的Python依赖包：

```bash
pip install -r requirements.txt
```

## 配置

服务器通过环境变量来配置。在项目根目录下创建一个名为 `.env` 的文件，并填入以下内容：

```
# 临时文件存储路径
MCP_SCRATCH=/tmp
```

## 运行服务器

使用以下命令启动服务器：

```bash
python server.py --host 0.0.0.0 --port 50002
```

## 主要功能 (Tools)

服务器以MCP工具的形式暴露了以下功能：

---

### 1. `Create_system`

根据 AMBER 的 `prmtop` 和 `inpcrd` 文件创建一个 OpenMM 系统。该功能会进行能量最小化，并将最终状态保存为 `.xml` 和 `.pdb` 文件。

- **参数:**
  - `prmtop_path` (str): AMBER `prmtop` 文件的路径。
  - `inpcrd_path` (str): AMBER `inpcrd` 文件的路径。
  - `temperature` (float, 可选, 默认 300.0): Langevin 积分器使用的温度（单位：K）。
  - `step_size` (float, 可选, 默认 0.004): 积分器的时间步长（单位：ps）。
- **返回:**
  - 成功时: `{"status": "success", "state_file": "/path/to/system_xxxxxx.xml", "pdb_file": "/path/to/system_xxxxxx.pdb"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 2. `heating_equilibration`

对一个已有的 OpenMM 系统（从 `.xml` 文件加载）进行加热和平衡。在模拟过程中，所有 CA 原子都会被约束。系统会逐渐加热到目标温度，然后进行平衡。

- **参数:**
  - `prmtop_path` (str): AMBER `prmtop` 文件的路径。
  - `state_file` (str): OpenMM 状态文件 (`.xml`) 的路径。
  - `temperature` (float, 可选, 默认 300.0): 目标温度（单位：K）。
  - `pressure` (float, 可选, 默认 1.0): 目标压力（单位：bar）。
  - `step_size` (float, 可选, 默认 0.004): 积分器的时间步长（单位：ps）。
  - `heating_time` (float, 可选, 默认 0.5): 加热阶段的时长（单位：ns）。
  - `eq_time` (float, 可选, 默认 1.0): 平衡阶段的时长（单位：ns）。
- **返回:**
  - 成功时: `{"status": "success", "state_file": "/path/to/equilibrated_xxxxxx.xml", "pdb_file": "/path/to/equilibrated_xxxxxx.pdb"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 3. `run_production_md`

对一个已有的 OpenMM 系统（从 `.xml` 文件加载）进行生产阶段的分子动力学模拟。

- **参数:**
  - `prmtop_path` (str): AMBER `prmtop` 文件的路径。
  - `state_file` (str): OpenMM 状态文件 (`.xml`) 的路径。
  - `temperature` (float, 可选, 默认 300.0): 目标温度（单位：K）。
  - `pressure` (float, 可选, 默认 1.0): 目标压力（单位：bar）。
  - `step_size` (float, 可选, 默认 0.004): 积分器的时间步长（单位：ps）。
  - `md_time` (float, 可选, 默认 10.0): 生产阶段 MD 的时长（单位：ns）。
  - `report_interval` (float, 可选, 默认 2): 报告进度的间隔（单位：ps）。
- **返回:**
  - 成功时: `{"status": "success", "trajectory_file": "...", "log_file": "...", "final_state_file": "...", "final_pdb_file": "..."}`
  - 失败时: `{"status": "error", "message": "error_details"}`

## 日志

服务器的运行日志会记录在 `logs/` 目录下，并按天保留。