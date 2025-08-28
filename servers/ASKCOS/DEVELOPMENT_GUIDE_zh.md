# AI4S-agent-tools 开发环境配置指南

## 📋 前置要求

- Python 3.10+
- uv (Python包管理器)
- Git

## 🚀 开发流程

### 1. 克隆项目并设置环境

```bash
# 克隆项目
git clone https://github.com/deepmodeling/AI4S-agent-tools.git
cd AI4S-agent-tools

# 安装 uv (如果还没有安装)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建新工具

```bash
# 复制模板
cp -r servers/_example servers/你的工具名称
cd servers/你的工具名称
```

### 3. 配置项目文件

#### 3.1 编辑 `pyproject.toml`

```toml:servers/你的工具名称/pyproject.toml
[project]
name = "your-tool-mcp-server"
version = "0.1.0"
description = "你的工具描述"
requires-python = ">=3.10"
dependencies = [
    "mcp",
    "fastmcp",
    "numpy",
    "scipy",
    # 添加你需要的其他依赖
]

[project.scripts]
your-tool-mcp = "server:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

#### 3.2 编辑 `metadata.json`

```json:servers/你的工具名称/metadata.json
{
    "name": "你的工具名称",
    "description": "你的工具科学计算描述",
    "author": "@你的GitHub用户名",
    "category": "materials",  // 选择合适的分类
    "transport": ["sse", "stdio"],
    "tools": ["你的函数1", "你的函数2"]
}
```

**可用分类：**
- `materials` - 材料科学
- `chemistry` - 化学
- `physics` - 物理
- `biology` - 生物学
- `research` - 研究工具
- `data` - 数据分析
- `machine-learning` - 机器学习
- `simulation` - 仿真
- `general` - 通用工具

#### 3.3 编写主服务器代码 `server.py`

```python:servers/你的工具名称/server.py
#!/usr/bin/env python3
"""
你的工具 MCP 服务器
描述你的工具的功能。
"""
import argparse
import os
from mcp.server.fastmcp import FastMCP

def parse_args():
    """解析MCP服务器的命令行参数。"""
    parser = argparse.ArgumentParser(description="你的工具 MCP 服务器")
    parser.add_argument('--port', type=int, default=50001, help='服务器端口 (默认: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机 (默认: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = FastMCP("你的工具名称", host=args.host, port=args.port)

# 定义你的工具函数
@mcp.tool()
def 你的函数1(param1: str, param2: float) -> dict:
    """
    你的函数描述。
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
        
    Returns:
        包含结果的字典
    """
    # 你的实现逻辑
    result = {
        "status": "success",
        "data": f"处理了 {param1} 和值 {param2}"
    }
    return result

@mcp.tool()
def 你的函数2(input_data: list) -> str:
    """
    另一个函数描述。
    
    Args:
        input_data: 输入数据列表
        
    Returns:
        处理结果字符串
    """
    # 你的实现逻辑
    return f"处理了 {len(input_data)} 个项目"

if __name__ == "__main__":
    # 从环境变量获取传输类型，默认为SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
```

### 4. 安装依赖并测试

```bash
# 安装依赖
uv sync

# 测试服务器（SSE模式）
python server.py --port 50001

# 测试服务器（stdio模式）
MCP_TRANSPORT=stdio python server.py
```

### 5. 创建MCP配置文件（可选）

```json:servers/你的工具名称/mcp-config.json
{
  "mcpServers": {
    "你的工具名称": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "servers/你的工具名称",
        "python",
        "server.py"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio"
      },
      "metadata": {
        "name": "你的工具名称",
        "description": "你的工具描述",
        "author": "@你的GitHub用户名",
        "category": "materials"
      }
    }
  }
}
```

### 6. 添加README文档

```markdown:servers/你的工具名称/README.md
# 你的工具名称

## 描述
简要描述你的工具的功能。

## 安装
```bash
cd servers/你的工具名称
uv sync
```

## 使用
```bash
# 运行服务器
python server.py --port 50001
```

## 可用工具
- `你的函数1`: 描述
- `你的函数2`: 描述

## 依赖
主要依赖及其用途列表。
```

### 7. 更新项目工具注册表

```bash
# 回到项目根目录
cd ../..

# 运行脚本更新工具注册表
python scripts/generate_tools_json.py
```

### 8. 提交代码

```bash
# 添加文件
git add servers/你的工具名称/
git add data/tools.json  # 如果运行了generate_tools_json.py

# 提交
git commit -m "添加新工具: 你的工具名称"

# 推送
git push origin main
```

## 🔧 开发技巧

### 调试模式
```bash
# 启用调试日志
python server.py --log-level DEBUG
```

### 环境变量
- `MCP_TRANSPORT`: 设置传输模式 (`sse` 或 `stdio`)
- `PORT`: 设置服务器端口

### 工具函数最佳实践
1. **类型注解**：为所有参数和返回值添加类型注解
2. **文档字符串**：详细描述函数功能、参数和返回值
3. **错误处理**：适当处理异常情况
4. **返回格式**：保持一致的返回数据格式

## 📚 参考资源

- [MCP协议文档](https://modelcontextprotocol.io/)
- [FastMCP文档](https://github.com/jlowin/fastmcp)
- [项目贡献指南](../../CONTRIBUTING.md)
- [工具展示页面](https://deepmodeling.github.io/AI4S-agent-tools/)

## 🤝 社区支持

遇到问题时可以：
1. 查看现有工具的实现作为参考
2. 在GitHub上提交Issue
3. 加入微信社区群讨论