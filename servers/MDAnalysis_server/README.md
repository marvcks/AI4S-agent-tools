# MDAnalysis MCP 服务器

这是一个使用 FastMCP 搭建的分子动力学分析服务器。它提供了一系列工具，用于分析分子动力学轨迹。

## 功能

服务器在 `server.py` 中定义了以下工具：

### `prepare_trajectories`

使用 MDAnalysis 准备和对齐轨迹。它会根据给定的选择（selection）将轨迹展开并对齐到第一帧。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径 (例如, Amber prmtop).
*   `trajectory_path` (str): 轨迹文件的路径 (例如, DCD, XTC).
*   `selection` (str): 用于对齐的原子选择字符串，使用 MDAnalysis 语法。默认为 "backbone"。

**返回值:**

*   `Dict[str, Any]`: 包含状态和对齐后的轨迹文件（XTC 格式）路径的字典。

### `calculate_rmsd`

使用 MDAnalysis 计算轨迹中某个选择相对于第一帧的 RMSD。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。
*   `selection` (str): 用于 RMSD 计算的原子选择字符串。默认为 "backbone"。

**返回值:**

*   `Dict[str, Any]`: 包含时间和 RMSD 数据的 npz 文件路径的字典。npz 文件包含 x (时间) 和 y (RMSD) 两个数组，单位为埃。

### `calculate_rmsf`

使用 MDAnalysis 计算蛋白质在轨迹上的 RMSF。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。

**返回值:**

*   `Dict[str, Any]`: 包含原子索引和 RMSF 数据的 npz 文件路径的字典。npz 文件包含 x (残基索引) 和 y (RMSF) 两个数组，单位为埃。

### `calculate_Rg`

使用 MDAnalysis 计算轨迹中某个选择的回旋半径 (Rg)。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。
*   `selection` (str): 用于 Rg 计算的原子选择字符串。默认为 "protein"。

**返回值:**

*   `Dict[str, Any]`: 包含时间和 Rg 数据的 npz 文件路径的字典。npz 文件包含 x (时间) 和 y (Rg) 两个数组，单位为埃。

### `calculate_distance`

使用 MDAnalysis 计算轨迹中两个选择之间的距离。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。
*   `selection1` (str): 第一个原子组的选择字符串。
*   `selection2` (str): 第二个原子组的选择字符串。

**返回值:**

*   `Dict[str, Any]`: 包含时间和距离数据的 npz 文件路径的字典。npz 文件包含 x (时间) 和 y (距离) 两个数组，单位为埃。

### `calculate_mm_gbsa`

使用 MMPBSA.py 计算轨迹的 MM-GBSA 结合自由能。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。
*   `ligand_resname` (str): 拓扑中配体的残基名称。默认为 "LIG"。
*   `interval` (int): 从轨迹中采样帧的间隔。默认为 1 (使用所有帧)。
*   `igb` (int): 使用的 GB 模型。默认为 5。
*   `salt_conc` (float): 盐浓度 (mol/L)。默认为 0.1。

**返回值:**

*   `Dict[str, Any]`: 包含状态、MM-GBSA 能量 (kcal/mol) 和标准差 (kcal/mol) 的字典。

### `export_molstar_html`

使用 molviewspec 导出一个可交互的 Mol* HTML 可视化文件。

**参数:**

*   `prmtop_path` (str): 拓扑文件的路径。
*   `trajectory_path` (str): 轨迹文件的路径。
*   `time` (float): 要可视化的轨迹时间点。
*   `selection` (str): 要导出的原子选择字符串。默认为 "not (resname HOH or resname WAT)"。

**返回值:**

*   `Dict[str, Any]`: 包含状态和生成的 Mol* HTML 文件路径的字典。

### `plot_picture`

从一个或多个包含 x 和 y 数据数组的 npz 文件生成图表。

**参数:**

*   `npz_path` (List[str]): npz 文件的路径列表。每个文件应包含 'x' 和 'y' 数组。
*   `Legend` (List[str]): 每个数据集的图例标签列表。如果为 None，则不显示图例。
*   `x_label` (str): x 轴的标签。默认为 "Time (ps)"。
*   `y_label` (str): y 轴的标签。默认为 "Value"。
*   `title` (str): 图表的标题。默认为 "Plot"。

**返回值:**

*   `Dict[str, Any]`: 包含状态和生成的图表图片（PNG 格式）路径的字典。

### `execute_python`

在受限环境中执行任意 Python 代码。

**参数:**

*   `code` (str): 要执行的 Python 代码。

**返回值:**

*   `dict`: 包含执行状态以及任何输出或错误消息的字典。

## 重要说明

*   服务器使用 `FastMCP` 框架。
*   核心分析功能依赖 `MDAnalysis` 库。
*   MM-GBSA 计算使用 `MMPBSA.py`。
*   3D 可视化使用 `molviewspec` 生成 Mol* 文件。
*   图表生成使用 `matplotlib`。
*   `execute_python` 函数包含一个安全检查，以防止不安全的操作。
*   服务器会将日志信息记录到 `logs/mcp_mdanalysis_{time}.log` 文件中。
*   临时文件存储在 `MCP_SCRATCH` 环境变量指定的目录中，默认为 `/tmp`。
