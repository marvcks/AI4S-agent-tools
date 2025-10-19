# UniDock2 MCP Server

本项目是一个基于 MCP (Multi-Computation Platform) 架构的分子对接服务器，后端使用 Uni-Dock2 工具。它提供了一系列用于分子对接前后处理和执行对接任务的工具接口。

## 功能

- 从 PDB 复合物中提取蛋白质和配体。
- 转换分子文件格式到 SDF。
- 计算对接盒子中心。
- 组合蛋白质和配体文件。
- 运行 Uni-Dock2 进行分子对接，支持常规对接和模板引导对接。

## 环境准备

在运行此服务器之前，您需要确保以下软件已经安装并配置在系统的环境变量中：

- **Uni-Dock2**: 分子对接工具。
- **Open Babel**: 用于分子文件格式转换。

## 安装

1.  克隆或下载本项目。
2.  安装所需的 Python 依赖包：

    ```bash
    pip install -r requirements.txt
    ```

## 运行服务

使用以下命令启动 MCP 服务器：

```bash
python server.py --host 0.0.0.0 --port 50004
```

服务器将在指定的地址和端口上启动，并开始监听请求。

---

## API 接口参考

以下是服务器提供的所有工具接口的详细说明。

### 1. `run_unidock2`

运行 Uni-Dock2 对接模拟。

- **参数**:
    - `receptor_pdb` (Path): 蛋白质受体 PDB 文件的路径。
    - `ligand_sdf` (Path): 小分子配体 SDF 文件的路径。
    - `center_x` (float): 对接盒子中心的 X 坐标。
    - `center_y` (float): 对接盒子中心的 Y 坐标。
    - `center_z` (float): 对接盒子中心的 Z 坐标。
    - `box_size_x` (float, 可选, 默认 30.0): 盒子在 X 轴的尺寸。
    - `box_size_y` (float, 可选, 默认 30.0): 盒子在 Y 轴的尺寸。
    - `box_size_z` (float, 可选, 默认 30.0): 盒子在 Z 轴的尺寸。
    - `template_sdf` (Path, 可选): 用于模板引导对接的参考配体 SDF 文件的路径。如果提供此项，将进行模板对接。

- **返回**:
    一个包含以下键的字典：
    - `status` (str): "success" 或 "error"。
    - `results_sdf` (Path): 对接结果 SDF 文件的路径。
    - `affinity` (List[float]): 识别出的亲和力分数列表。

### 2. `extract_template_ligand`

从包含蛋白质-配体复合物的 PDB 文件中提取天然配体和蛋白质。

- **参数**:
    - `holo_pdb` (Path): 包含蛋白质和配体的 PDB 文件的路径。
    - `ligand_resname` (str, 可选, 默认 "LIG"): PDB 文件中配体的残基名称。

- **返回**:
    一个包含以下键的字典：
    - `status` (str): "success" 或 "error"。
    - `receptor_pdb` (Path): 生成的仅包含蛋白质的 PDB 文件的路径。
    - `ligand_sdf` (Path): 生成的配体 SDF 文件的路径。

### 3. `calculate_box_center`

计算 PDB 文件中指定区域的几何中心。

- **参数**:
    - `receptor_pdb` (Path): 蛋白质受体 PDB 文件的路径。
    - `selection` (str): MDAnalysis 的选择字符串，用于定义计算中心的原子区域 (例如, "resid 100 and name CA")。

- **返回**:
    一个包含以下键的字典：
    - `status` (str): "success" 或 "error"。
    - `center` (Tuple[float, float, float]): 计算出的几何中心 (x, y, z) 坐标。

### 4. `combine_protein_ligand`

将一个蛋白质 PDB 文件和一个配体 SDF 文件合并成一个单独的复合物 PDB 文件。

- **参数**:
    - `receptor_pdb` (Path): 蛋白质受体 PDB 文件的路径。
    - `ligand_sdf` (Path): 配体 SDF 文件的路径。
    - `ligand_resname` (str, 可选, 默认 "LIG"): 在合并后的 PDB 文件中分配给配体的残基名称。

- **返回**:
    一个包含以下键的字典：
    - `status` (str): "success" 或 "error"。
    - `complex_pdb` (Path): 生成的复合物 PDB 文件的路径。

### 5. `convert_file_to_sdf`

使用 Open Babel 将一个分子结构文件（如 PDB, MOL2）转换为 SDF 格式。

- **参数**:
    - `input_file` (Path): 输入的分子结构文件的路径。

- **返回**:
    一个包含以下键的字典：
    - `status` (str): "success" 或 "error"。
    - `sdf_file` (Path): 转换后的 SDF 文件的路径。
