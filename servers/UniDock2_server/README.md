# UniDock2 MCP 服务器

这是一个使用 CalculationMCP 框架为 Uni-Dock2 提供分子对接服务的MCP服务器。

## 功能

此服务器提供了一系列用于分子对接前后处理的工具，主要通过 `server.py` 脚本提供。

### 主要函数

- **`extract_template_ligand(holo_pdb: str, ligand_resname: str="LIG")`**
  - **功能**: 从一个包含受体和配体的PDB文件（holo_pdb）中提取天然配体。
  - **参数**:
    - `holo_pdb` (str): 包含受体和配体的PDB文件的路径。
    - `ligand_resname` (str, 可选): PDB文件中配体的残基名称，默认为 "LIG"。
  - **返回值**: 一个字典，包含状态信息、仅含受体的PDB文件路径和仅含配体的SDF文件路径。

- **`convert_file_to_sdf(input_file: str)`**
  - **功能**: 使用 Open Babel 将分子结构文件（PDB, MOL2, SDF）转换为SDF格式。
  - **参数**:
    - `input_file` (str): 输入结构文件的路径。
  - **返回值**: 一个字典，包含状态信息和转换后的SDF文件的路径。

- **`combine_protein_ligand(receptor_pdb: str, ligand_sdf: str, ligand_resname: str="LIG")`**
  - **功能**: 将一个受体PDB文件和一个配体SDF文件合并成一个单独的PDB文件，以便进行后续的分子动力学模拟。
  - **参数**:
    - `receptor_pdb` (str): 受体PDB文件的路径。
    - `ligand_sdf` (str): 配体SDF文件的路径。
    - `ligand_resname` (str, 可选): 在合并后的PDB文件中分配给配体的残基名称，默认为 "LIG"。
  - **返回值**: 一个字典，包含状态信息和合并后的PDB文件的路径。

- **`run_unidock2(...)`**
  - **功能**: 运行 Uni-Dock2 对接模拟。
  - **参数**:
    - `receptor_pdb` (str): 受体PDB文件的路径。
    - `ligand_sdf` (str): 配体SDF文件的路径。
    - `center_x`, `center_y`, `center_z` (float): 对接盒子中心的X, Y, Z坐标。
    - `box_size_x`, `box_size_y`, `box_size_z` (float, 可选): 对接盒子在X, Y, Z维度的大小，默认为30.0。
    - `template_sdf` (Optional[str], 可选): 用于引导对接的模板配体SDF文件的路径，默认为 None。
  - **返回值**: 一个字典，包含状态信息和对接结果文件的路径。

### 重要提示

- **依赖关系**:
  - 本项目依赖于 `MDAnalysis`, `nanoid`, `obabel` (Open Babel), 和 `unidock2`。请确保这些依赖项已正确安装。
- **环境变量**:
  - `MCP_SCRATCH`: 用于存储临时文件的目录，默认为 `/tmp`。
- **日志**:
  - 服务器日志会记录在 `logs/` 目录下，并每天轮换。

## 如何运行

通过以下命令启动服务器：

```bash
python server.py --port 50004
```

服务器将在 `0.0.0.0:50004` 上启动。
