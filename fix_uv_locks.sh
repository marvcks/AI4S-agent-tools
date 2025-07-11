#!/bin/bash
# 设置使用系统 Python
export UV_PYTHON_PREFERENCE=system

# 需要修复的服务器列表
servers=("Paper_Search" "DPACalculator" "deepmd_docs_rag" "pubchem")

for server in "${servers[@]}"; do
    echo "修复 $server..."
    cd "/Users/lhappy/workbench/AI4S-agent-tools/servers/$server"
    
    
    # 生成新的 uv.lock
    uv lock
    
    # 同步安装
    uv sync
    
    echo "$server 完成！"
    echo "-------------------"
done