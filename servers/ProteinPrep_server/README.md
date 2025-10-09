# ProteinPrep 服务器

本项目是一个用于分子动力学（MD）模拟中蛋白质结构预处理的MCP（Modular Compute Platform）服务器。它集成了一系列工具，可以自动化地从PDB数据库获取结构、修复蛋白质（如添加缺失的残基和原子）、处理配体、并最终生成适用于AMBER等模拟软件的拓扑和坐标文件。

## 安装

安装所有必需的Python依赖包：

```bash
pip install -r requirements.txt
```

此外，本服务器依赖于AmberTools的命令行工具（`pdb4amber`, `antechamber`, `parmchk2`, `tleap`, `parmed`）。请确保AmberTools已正确安装并配置在系统的`PATH`环境变量中。

## 配置

服务器通过环境变量来配置与Cloudflare R2对象存储的连接，用于存储和分享生成的文件。

在项目根目录下创建一个名为 `.env` 的文件，并填入以下内容：

```
# Cloudflare R2 配置
ACCESS_KEY_ID=your_r2_access_key_id
ACCOUNT_ID=your_r2_account_id
SECRET_ACCESS_KEY=your_r2_secret_access_key
BUCKET_NAME=your_r2_bucket_name
ENDPOINT_URL=https://<your_account_id>.r2.cloudflarestorage.com

# 临时文件存储路径
MCP_SCRATCH=/tmp
```

请将 `your_*` 替换为您的实际Cloudflare R2凭证。

## 运行服务器

使用以下命令启动服务器：

```bash
python server.py
```

您也可以指定主机和端口：

```bash
python server.py --host 0.0.0.0 --port 50001
```

## 主要功能 (Tools)

服务器以MCP工具的形式暴露了以下功能：

---

### 1. `fetch_rcsb`

从RCSB PDB数据库获取指定的蛋白质结构文件。

- **参数:**
  - `pdb_id` (str): 要获取的蛋白质的4位PDB ID。
- **返回:**
  - 成功时: `{"status": "success", "pdb_path": "/path/to/pdb_id.pdb"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 2. `Protein_Prep`

对输入的PDB文件进行预处理，为MD模拟做准备。该工具会执行以下操作：
1.  移除所有已存在的氢原子。
2.  寻找并添加缺失的残基（不包括链的末端）。
3.  替换非标准的氨基酸残基。
4.  寻找并添加缺失的原子。
5.  根据设定的pH值添加氢原子。
6.  删除用户指定的残基（如水分子、配体等）。
7.  调用 `pdb4amber` 工具进行最终处理，使其与AMBER力场兼容。

- **参数:**
  - `pdb_path` (str): 输入的PDB文件路径。
  - `ph` (float, 可选, 默认 7.0): 用于确定质子化状态的pH值。
  - `toDeleteRes` (List[str], 可选): 需要从结构中删除的残基名称列表（例如 `["PO4", "BEM"]`）。
- **返回:**
  - 成功时: `{"status": "success", "prepared_pdb_path": "/path/to/protein_fixer_amber.pdb"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 3. `get_protein_sequence`

从PDB文件中提取氨基酸序列。

- **参数:**
  - `pdb_path` (str): 输入的PDB文件路径。
- **返回:**
  - 成功时: `{"status": "success", "sequence": "CHAIN_A_SEQ:CHAIN_B_SEQ"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 4. `parametrize_ligand`

使用AmberTools中的 `antechamber` 和 `parmchk2` 为小分子配体生成力场参数。

- **参数:**
  - `pdb_path` (str): 包含配体的PDB文件路径。
  - `ligand_resname` (str, 可选, 默认 "LIG"): PDB文件中配体的残基名称。
  - `charge` (int, 可选, 默认 0): 配体的净电荷。
- **返回:**
  - 成功时: `{"status": "success", "ligand_id": "xxxxxx", "mol2_path": "...", "frcmod_path": "..."}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 5. `run_tleap`

运行 `tleap` 来为蛋白质（或蛋白质-配体复合物）系统生成拓扑（prmtop）和坐标（inpcrd）文件。此过程包括：
1.  加载蛋白质力场（ff14SB）、GAFF（用于配体）和水模型（TIP3P）。
2.  如果提供了配体信息，则加载配体的 `mol2` 和 `frcmod` 文件。
3.  加载蛋白质结构，并使用TIP3PBOX水盒子进行溶剂化。
4.  添加抗衡离子以中和系统。
5.  保存溶剂化后的系统拓扑和坐标文件。
6.  应用氢质量重分配（HMR）以允许在模拟中使用更长的时间步长。

- **参数:**
  - `prepared_pdb_path` (str): 经过 `Protein_Prep` 处理后的PDB文件路径。
  - `ligand_res_name` (str): 配体的残基名称。如果系统中没有配体，则留空。
  - `ligand_id` (str): 从 `parametrize_ligand` 返回的配体ID。如果系统中没有配体，则留空。
- **返回:**
  - 成功时: `{"status": "success", "prmtop_path": "/path/to/protein_solv_hmr.parm7", "inpcrd_path": "/path/to/protein_solv_hmr.rst7"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

---

### 6. `local_file_to_r2_url`

将服务器本地的任何文件上传到配置好的Cloudflare R2存储桶，并返回一个公开可访问的URL。

- **参数:**
  - `local_file` (str): 要上传的本地文件的路径。
- **返回:**
  - 成功时: `{"status": "success", "url": "https://pyscftoolmcp.cc/filename"}`
  - 失败时: `{"status": "error", "message": "error_details"}`

## 日志

服务器的运行日志会记录在 `logs/` 目录下，并按天保留。日志级别可以通过命令行参数 `--log-level` 进行设置。
