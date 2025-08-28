# 如何将 ASKCOS 集成为 AI4S-agent-tools 的工具

本文档旨在指导开发者如何将 ASKCOS 的功能封装成一个 agent-tool，并集成到 `AI4S-agent-tools` 项目中。

## 1. 前置要求：部署 ASKCOS

在开始开发 agent-tool 之前，您**必须**拥有一套正在运行的 ASKCOS 服务。

1.  **硬件/软件准备**: 准备一台满足 ASKCOS 要求的服务器（x86 CPU, >= 32 GB RAM, Ubuntu >= 20.04, Docker）。<mcreference link="https://askcos-docs.mit.edu/guide/1-Introduction/1.1-Introduction.html" index="0">0</mcreference>
2.  **部署 ASKCOS**:
    -   根据 [ASKCOS 官方文档](https://askcos-docs.mit.edu/guide/1-Introduction/1.1-Introduction.html) 的指引，克隆其代码库并执行 `make deploy`。
    -   部署成功后，确保您可以通过浏览器访问其 Web UI，并通过 `http://<your-askcos-ip>:9100/docs` 查看其 FastAPI 交互式 API 文档。<mcreference link="https://askcos-docs.mit.edu/guide/1-Introduction/1.1-Introduction.html" index="0">0</mcreference>
3.  **获取 API 地址**: 记下 ASKCOS 后端 API 的地址（例如 `http://127.0.0.1:9100`）。我们的 agent-tool 将会向这个地址发送请求。

## 2. 开发流程

### 第一步：创建工具目录

在 `AI4S-agent-tools/servers/` 目录下创建一个新的文件夹，用于存放您的工具。

```bash
mkdir /home/jiaodu/projects/AI4S-agent-tools/servers/askcos_tool
cd /home/jiaodu/projects/AI4S-agent-tools/servers/askcos_tool
```

### 第二步：配置 `pyproject.toml`

创建 `pyproject.toml` 文件，声明项目的基本信息和依赖。对于调用 API 的工具，`requests` 是必不可少的依赖。

```toml:%2Fhome%2Fjiaodu%2Fprojects%2FAI4S-agent-tools%2Fservers%2Faskcos_tool%2Fpyproject.toml
[project]
name = "askcos_tool"
version = "0.1.0"
description = "A tool to perform retrosynthesis planning using the ASKCOS API."
dependencies = [
    "fastmcp",
    "requests"
]
```

### 第三步：配置 `metadata.json`

创建 `metadata.json` 文件，用于向 `AI4S-agent-tools` 注册您的工具和它提供的功能。

```json:%2Fhome%2Fjiaodu%2Fprojects%2FAI4S-agent-tools%2Fservers%2Faskcos_tool%2Fmetadata.json
{
    "name": "ASKCOS Tool",
    "description": "Interfaces with an ASKCOS instance to perform computer-aided synthesis planning.",
    "author": "Your Name",
    "category": "chemistry",
    "transport": ["stdio", "sse"],
    "tools": {
        "retrosynthesis": {
            "name": "retrosynthesis_planning",
            "description": "Performs retrosynthesis planning for a given molecule SMILES.",
            "inputs": {
                "smiles": {
                    "type": "string",
                    "description": "The SMILES string of the target molecule."
                }
            },
            "outputs": {
                "result": {
                    "type": "object",
                    "description": "The synthesis plan returned by ASKCOS."
                }
            }
        }
    }
}
```

### 第四步：编写主服务器代码 `server.py`

这是工具的核心。它将使用 `requests` 库来调用您部署的 ASKCOS 实例的 API。

**注意**: 您需要根据 ASKCOS 的实际 API 文档来确定请求的 URL、方法（GET/POST）、参数和返回的数据结构。以下是一个示例，演示了如何封装一个假设的逆合成分析 API。

```python:%2Fhome%2Fjiaodu%2Fprojects%2FAI4S-agent-tools%2Fservers%2Faskcos_tool%2Fserver.py
import os
from fastmcp import FastMCP, Tool, tool
import requests

# 从环境变量或配置文件中获取 ASKCOS API 的地址
ASKCOS_API_URL = os.environ.get("ASKCOS_API_URL", "http://127.0.0.1:9100")

class ASKCOSAPIWrapper(Tool):
    @tool
    def retrosynthesis_planning(self, smiles: str) -> dict:
        """
        Performs retrosynthesis planning for a given molecule SMILES.

        This function sends a request to the ASKCOS API's retrosynthesis
        endpoint and returns the resulting synthesis plan.

        :param smiles: The SMILES string of the target molecule.
        :return: A dictionary containing the synthesis plan.
        """
        # 注意：这里的 endpoint (`/api/retrosynthesis`) 是一个示例，
        # 你需要参考你部署的 ASKCOS 实例的 API 文档 (`/docs`) 来确定确切的路径和参数。
        endpoint = f"{ASKCOS_API_URL}/api/v2/tree-builder/plan"
        
        # ASKCOS 的 API 可能需要用户认证或特定的请求体格式
        # 这里是一个简化的 POST 请求示例
        payload = {
            "smiles": smiles,
            "max_depth": 5,
            # 其他参数...
        }
        
        try:
            # 你可能需要处理登录和 session
            # response = requests.post(endpoint, json=payload, auth=('user', 'pass'))
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # 如果请求失败 (状态码 4xx 或 5xx), 则抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to ASKCOS API: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

if __name__ == "__main__":
    mcp_server = FastMCP(
        tools=[ASKCOSAPIWrapper()],
        title="ASKCOS Tool Server",
        description="A server for performing actions on an ASKCOS instance.",
    )
    mcp_server.run()
```

### 第五步：安装依赖与测试

1.  **安装依赖**:
    ```bash
    # 激活你的虚拟环境
    # source .../bin/activate
    pip install -e .
    ```
2.  **本地测试**:
    在启动服务器前，先设置环境变量指向你的 ASKCOS API 地址。
    ```bash
    export ASKCOS_API_URL="http://<your-askcos-ip>:9100"
    python server.py --transport stdio
    ```
    然后，您可以从另一个终端向它发送 MCP 协议格式的 JSON 请求来测试功能。

### 第六步：更新项目工具注册表

最后，运行项目根目录下的脚本，将您的新工具添加到 `data/tools.json` 中。

```bash
cd /home/jiaodu/projects/AI4S-agent-tools/
python scripts/generate_tools_json.py
```

完成以上步骤后，您的 `askcos_tool` 就成功集成到 `AI4S-agent-tools` 项目中了。
