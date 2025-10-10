# UniDock2 MCP 服务器

这是一个使用 CalculationMCP 框架为 Uni-Dock2 提供分子对接服务的MCP服务器。它集成了一系列工具，可以自动化地进行分子对接前后的处理。

## 安装

安装所有必需的Python依赖包。假设依赖项保存在 `requirements.txt` 中：

```bash
pip install -r requirements.txt
```

此外，本服务器依赖于 `unidock2` 和 `obabel` 命令行工具。请确保他们已正确安装并配置在系统的`PATH`环境变量中。

## 配置

服务器通过环境变量来配置。例如，可以设置临时文件存储路径。

在项目根目录下创建一个名为 `.env` 的文件，并填入以下内容：

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
python server.py --host 0.0.0.0 --port 50004
```

## 主要功能 (Tools)

服务器以MCP工具的形式暴露了以下功能：

---

### 1. `extract_template_ligand`

从一个包含受体和配体的PDB文件（holo_pdb）中提取天然配体。

- **参数:**
  - `holo_pdb` (str): 包含受体和配体的PDB文件的路径。
  - `ligand_resname` (str, 可选, 默认 "LIG"): PDB文件中配体的残基名称。
- **返回:**
  - 成功时: `{"status": "success", "receptor_pdb_path": "/path/to/receptor.pdb", "ligand_sdf_path": "/path/to/ligand.sdf"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 2. `convert_file_to_sdf`

使用 Open Babel 将分子结构文件（PDB, MOL2, SDF）转换为SDF格式。

- **参数:**
  - `input_file` (str): 输入结构文件的路径。
- **返回:**
  - 成功时: `{"status": "success", "output_sdf_path": "/path/to/output.sdf"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 3. `combine_protein_ligand`

将一个受体PDB文件和一个配体SDF文件合并成一个单独的PDB文件，以便进行后续的分子动力学模拟。

- **参数:**
  - `receptor_pdb` (str): 受体PDB文件的路径。
  - `ligand_sdf` (str): 配体SDF文件的路径。
  - `ligand_resname` (str, 可选, 默认 "LIG"): 在合并后的PDB文件中分配给配体的残基名称。
- **返回:**
  - 成功时: `{"status": "success", "combined_pdb_path": "/path/to/combined.pdb"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 4. `run_unidock2`

运行 Uni-Dock2 对接模拟。

- **参数:**
  - `receptor_pdb` (str): 受体PDB文件的路径。
  - `ligand_sdf` (str): 配体SDF文件的路径。
  - `center_x` (float): 对接盒子中心的X坐标。
  - `center_y` (float): 对接盒子中心的Y坐标。
  - `center_z` (float): 对接盒子中心的Z坐标。
  - `box_size_x` (float, 可选, 默认 30.0): 对接盒子在X维度的大小。
  - `box_size_y` (float, 可选, 默认 30.0): 对接盒子在Y维度的大小。
  - `box_size_z` (float, 可选, 默认 30.0): 对接盒子在Z维度的大小。
  - `template_sdf` (str, 可选): 用于引导对接的模板配体SDF文件的路径。
- **返回:**
  - 成功时: `{"status": "success", "docking_result_path": "/path/to/docking_results.sdf"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

## 日志

服务器的运行日志会记录在 `logs/` 目录下，并按天保留。