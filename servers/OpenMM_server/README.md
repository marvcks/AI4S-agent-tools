# OpenMM MCP 服务器

本项目提供了一个基于 MCP (Molecule Computing Platform) 的服务器，用于使用 OpenMM 进行分子动力学模拟。

## 环境设置

1.  **安装依赖**:
    确保已经安装 `dotenv`, `loguru`, `openmm`, `nanoid` 等必要的 Python 库。

2.  **环境变量**:
    项目使用 `.env` 文件来配置环境变量。`MCP_SCRATCH` 变量定义了临时文件的存储路径，默认为 `/tmp`。

## 如何运行

通过以下命令启动服务器：

```bash
python server.py --host 0.0.0.0 --port 50002
```

## 可用功能

服务器提供了以下几个功能，可以通过 MCP 客户端调用。

### 1. `Create_system`

根据 AMBER 的 `prmtop` 和 `inpcrd` 文件创建一个 OpenMM 系统。该功能会进行能量最小化，并将最终状态保存为 `.xml` 和 `.pdb` 文件。

**参数:**

*   `prmtop_path` (str): AMBER `prmtop` 文件的路径。
*   `inpcrd_path` (str): AMBER `inpcrd` 文件的路径。
*   `temperature` (float, 可选): Langevin 积分器使用的温度（单位：K），默认为 300.0 K。
*   `step_size` (float, 可选): 积分器的时间步长（单位：ps），默认为 0.004 ps。

**返回值:**

一个包含操作状态和生成文件路径的字典。

```json
{
  "status": "success",
  "state_file": "/path/to/system_xxxxxx.xml",
  "pdb_file": "/path/to/system_xxxxxx.pdb"
}
```

### 2. `heating_equilibration`

对一个已有的 OpenMM 系统（从 `.xml` 文件加载）进行加热和平衡。在模拟过程中，所有 CA 原子都会被约束。系统会逐渐加热到目标温度，然后进行平衡。

**参数:**

*   `prmtop_path` (str): AMBER `prmtop` 文件的路径。
*   `state_file` (str): OpenMM 状态文件 (`.xml`) 的路径。
*   `temperature` (float, 可选): 目标温度（单位：K），默认为 300.0 K。
*   `pressure` (float, 可选): 目标压力（单位：bar），默认为 1.0 bar。
*   `step_size` (float, 可选): 积分器的时间步长（单位：ps），默认为 0.004 ps。
*   `heating_time` (float, 可选): 加热阶段的时长（单位：ns），默认为 0.5 ns。
*   `eq_time` (float, 可选): 平衡阶段的时长（单位：ns），默认为 1.0 ns。

**返回值:**

一个包含操作状态和最终文件路径的字典。

```json
{
  "status": "success",
  "state_file": "/path/to/equilibrated_xxxxxx.xml",
  "pdb_file": "/path/to/equilibrated_xxxxxx.pdb"
}
```

### 3. `run_production_md`

对一个已有的 OpenMM 系统（从 `.xml` 文件加载）进行生产阶段的分子动力学模拟。

**参数:**

*   `prmtop_path` (str): AMBER `prmtop` 文件的路径。
*   `state_file` (str): OpenMM 状态文件 (`.xml`) 的路径。
*   `temperature` (float, 可选): 目标温度（单位：K），默认为 300.0 K。
*   `pressure` (float, 可选): 目标压力（单位：bar），默认为 1.0 bar。
*   `step_size` (float, 可选): 积分器的时间步长（单位：ps），默认为 0.004 ps。
*   `md_time` (float, 可选): 生产阶段 MD 的时长（单位：ns），默认为 10.0 ns。
*   `report_interval` (float, 可选): 报告进度的间隔（单位：ps），默认为 2 ps。

**返回值:**

一个包含操作状态和生成文件路径的字典。

```json
{
  "status": "success",
  "trajectory_file": "/path/to/md_trajectory_xxxxxx.xtc",
  "log_file": "/path/to/md_log_xxxxxx.log",
  "final_state_file": "/path/to/md_final_state_xxxxxx.xml",
  "final_pdb_file": "/path/to/md_final_structure_xxxxxx.pdb"
}
```
