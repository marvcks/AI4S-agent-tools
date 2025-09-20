## 环境安装

```bash
pip install -r requirements.txt


# 安装 xTB
export PATH=/xxx/xtb-dist/bin:$PATH
cp /xxx/xtb-dist/bin /opt/orca504/orca_5_0_4_linux_x86-64_shared_openmpi411/otool_xtb

# 安装 autodE
git clone https://github.com/duartegroup/autodE.git
cd autodE/
pip install -e .


# 安装 packmol
# 从这里 https://github.com/m3g/packmol/releases/tag/v21.1.0 下载 
export PATH=/xxxx/packmol-21.1.0:$PATH

# 设置api for mannual RAG
export DASHSCOPE_API_KEY="sk-xxxxxxxxx"

python server.py
```

## 文件说明

`vector_db_orca_manual_qwen` 是ORCA手册的Vector Database，用RAG.

`test_data` 中存放了便于直接测试MCP工具的文件.

