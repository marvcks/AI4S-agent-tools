# MDAnalysis MCP 服务器

这是一个使用 FastMCP 搭建的分子动力学分析服务器。它提供了一系列工具，用于分析分子动力学轨迹。

## 安装

安装所有必需的Python依赖包：

```bash
pip install -r requirements.txt
```

此外，本服务器依赖于 AmberTools 的 `MMPBSA.py`。请确保 AmberTools 已正确安装并配置在系统的`PATH`环境变量中。

## 配置

服务器通过环境变量来配置。在项目根目录下创建一个名为 `.env` 的文件，并填入以下内容：

```
# 临时文件存储路径
MCP_SCRATCH=/tmp
```

## 运行服务器

使用以下命令启动服务器：

```bash
python server.py
```

您也可以指定主机和端口：

```bash
python server.py --host 0.0.0.0 --port <your_port>
```

## 主要功能 (Tools)

服务器以MCP工具的形式暴露了以下功能：

---

### 1. `prepare_trajectories`

使用 MDAnalysis 准备和对齐轨迹。它会根据给定的选择（selection）将轨迹展开并对齐到第一帧。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径 (例如, Amber prmtop).
  - `trajectory_path` (str): 轨迹文件的路径 (例如, DCD, XTC).
  - `selection` (str, 可选, 默认 "backbone"): 用于对齐的原子选择字符串，使用 MDAnalysis 语法。
- **返回:**
  - 成功时: `{"status": "success", "aligned_trajectory_path": "/path/to/aligned.xtc"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 2. `calculate_rmsd`

使用 MDAnalysis 计算轨迹中某个选择相对于第一帧的 RMSD。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
  - `selection` (str, 可选, 默认 "backbone"): 用于 RMSD 计算的原子选择字符串。
- **返回:**
  - 成功时: `{"status": "success", "npz_path": "/path/to/rmsd.npz"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 3. `calculate_rmsf`

使用 MDAnalysis 计算蛋白质在轨迹上的 RMSF。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
- **返回:**
  - 成功时: `{"status": "success", "npz_path": "/path/to/rmsf.npz"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 4. `calculate_Rg`

使用 MDAnalysis 计算轨迹中某个选择的回旋半径 (Rg)。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
  - `selection` (str, 可选, 默认 "protein"): 用于 Rg 计算的原子选择字符串。
- **返回:**
  - 成功时: `{"status": "success", "npz_path": "/path/to/rg.npz"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 5. `calculate_distance`

使用 MDAnalysis 计算轨迹中两个选择之间的距离。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
  - `selection1` (str): 第一个原子组的选择字符串。
  - `selection2` (str): 第二个原子组的选择字符串。
- **返回:**
  - 成功时: `{"status": "success", "npz_path": "/path/to/distance.npz"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 6. `calculate_mm_gbsa`

使用 MMPBSA.py 计算轨迹的 MM-GBSA 结合自由能。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
  - `ligand_resname` (str, 可选, 默认 "LIG"): 拓扑中配体的残基名称。
  - `interval` (int, 可选, 默认 1): 从轨迹中采样帧的间隔。
  - `igb` (int, 可选, 默认 5): 使用的 GB 模型。
  - `salt_conc` (float, 可选, 默认 0.1): 盐浓度 (mol/L)。
- **返回:**
  - 成功时: `{"status": "success", "energy": -12.34, "std_dev": 1.23}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 7. `export_molstar_html`

使用 molviewspec 导出一个可交互的 Mol* HTML 可视化文件。

- **参数:**
  - `prmtop_path` (str): 拓扑文件的路径。
  - `trajectory_path` (str): 轨迹文件的路径。
  - `time` (float): 要可视化的轨迹时间点。
  - `selection` (str, 可选, 默认 "not (resname HOH or resname WAT)"): 要导出的原子选择字符串。
- **返回:**
  - 成功时: `{"status": "success", "html_path": "/path/to/molstar.html"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 8. `plot_picture`

从一个或多个包含 x 和 y 数据数组的 npz 文件生成图表。

- **参数:**
  - `npz_path` (List[str]): npz 文件的路径列表。
  - `Legend` (List[str], 可选): 每个数据集的图例标签列表。
  - `x_label` (str, 可选, 默认 "Time (ps)"): x 轴的标签。
  - `y_label` (str, 可选, 默认 "Value"): y 轴的标签。
  - `title` (str, 可选, 默认 "Plot"): 图表的标题。
- **返回:**
  - 成功时: `{"status": "success", "plot_path": "/path/to/plot.png"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 9. `execute_python`

在受限环境中执行任意 Python 代码。

- **参数:**
  - `code` (str): 要执行的 Python 代码。
- **返回:**
  - 成功时: `{"status": "success", "output": "..."}`
  - 失败时: `{"status": "error", "message": "error_details"}`

## 日志

服务器的运行日志会记录在 `logs/` 目录下，并按天保留。