# 贡献指南 | Contributing Guide

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 🇨🇳 中文

### 欢迎贡献！

AI4S-agent-tools 是一个由 DeepModeling 社区维护的开源项目，旨在为科学研究构建智能代理工具库。我们欢迎所有形式的贡献！

### 📋 贡献前准备

1. **Fork 仓库** - 点击右上角的 Fork 按钮
2. **克隆到本地**
   ```bash
   git clone https://github.com/你的用户名/AI4S-agent-tools.git
   cd AI4S-agent-tools
   ```
3. **安装 UV** (Python 依赖管理工具)
   ```bash
   pip install uv
   ```
4. **了解项目结构**
   ```
   AI4S-agent-tools/
   ├── servers/           # 所有 MCP 服务器
   │   ├── _example/      # 示例模板
   │   └── your_tool/     # 你的新工具
   ├── scripts/           # 工具脚本
   ├── config/            # 配置文件
   ├── showcase/          # 展示页面（自动生成）
   └── TOOLS.json         # 工具注册表（自动生成）
   ```

### 🚀 快速开始：添加新工具

#### 方法一：使用模板（推荐）

1. **复制示例服务器**
   ```bash
   cp -r servers/_example servers/你的工具名称
   ```

2. **修改 server.py**
   ```python
   from servers.server_utils import mcp_server, setup_server
   from mcp.server.fastmcp import FastMCP

   @mcp_server("你的工具名称", "工具的清晰描述", author="@你的GitHub用户名", category="分类名称")
   def create_server(host="0.0.0.0", port=50001):
       mcp = FastMCP("你的工具名称", host=host, port=port)
       
       @mcp.tool()
       def 你的函数名(参数: str) -> dict:
           """函数功能说明"""
           try:
               # 实现你的科学计算逻辑
               return {"result": "结果"}
           except Exception as e:
               return {"error": f"失败: {str(e)}"}
       
       return mcp

   if __name__ == "__main__":
       setup_server().run()
   ```

3. **更新依赖配置** (`pyproject.toml`)
   ```toml
   [project]
   name = "你的工具名称"
   version = "0.1.0"
   description = "工具描述"
   requires-python = ">=3.8"
   dependencies = [
       "fastmcp>=0.5.0",
       # 添加你需要的其他依赖
   ]
   ```

4. **添加说明文档** (`README.md`)
   ```markdown
   # 你的工具名称

   ## 功能介绍
   详细说明你的工具能做什么...

   ## 使用示例
   ```python
   # 展示如何使用你的工具
   ```

   ## 安装和运行
   ```bash
   cd servers/你的工具名称
   uv sync
   python server.py --port 50001
   ```
   ```

5. **选择正确的分类**
   
   在 `@mcp_server` 装饰器中使用以下分类之一：
   - `"Materials Science"` - 材料科学相关工具
   - `"Chemistry"` - 化学计算和分析
   - `"Biology"` - 生物系统分析
   - `"Physics"` - 物理模拟和计算
   - `"Research Tools"` - 文献搜索和知识管理
   - `"Simulation"` - 分子动力学和建模
   - `"Data & Analysis"` - 数据处理和可视化
   - `"Machine Learning"` - AI/ML 模型
   - `"General Tools"` - 通用工具

6. **更新工具注册表**
   ```bash
   python scripts/generate_tools_json.py
   ```

### 📝 代码规范

1. **遵循现有模式**
   - 使用 `@mcp_server` 装饰器注册服务器
   - 使用 `@mcp.tool()` 装饰器定义工具函数
   - 始终包含错误处理
   - 在 `@mcp_server` 中指定正确的分类（category）

2. **命名规范**
   - 服务器目录名：小写，使用下划线分隔
   - Python 文件：遵循 PEP 8 规范
   - 工具函数：清晰描述功能的动词短语

3. **文档要求**
   - 每个工具必须有 README.md
   - 函数必须有 docstring
   - 复杂功能需要使用示例

4. **测试要求**
   - 手动测试所有功能
   - 提供测试命令和预期结果
   - 确保服务器能正常启动

### 🔧 开发流程

1. **创建功能分支**
   ```bash
   git checkout -b feature/你的功能名称
   ```

2. **开发和测试**
   ```bash
   # 安装依赖
   cd servers/你的工具名称
   uv sync
   
   # 运行服务器
   python server.py --port 50001 --log-level DEBUG
   
   # 检查日志
   tail -f ~/.你的工具名称/*.log
   ```

3. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 添加新工具 - 简短描述"
   ```

4. **推送并创建 PR**
   ```bash
   git push origin feature/你的功能名称
   ```

### ⚠️ 注意事项

- **端口分配**：检查 TOOLS.json 避免端口冲突
- **依赖管理**：使用 UV 管理依赖，不要直接修改 uv.lock
- **安全性**：不要提交密钥或敏感信息
- **兼容性**：确保 Python >= 3.8
- **展示页面**：你的工具会自动出现在 [项目展示页面](https://lhhhappy.github.io/AI4S-agent-tools/)

### 🎯 PR 检查清单

提交 PR 前请确认：

- [ ] 代码能正常运行
- [ ] 已添加必要的文档
- [ ] 已运行 `generate_tools_json.py`
- [ ] 已测试主要功能
- [ ] 代码风格一致
- [ ] 没有硬编码的路径或密钥

---

<a name="english"></a>
## 🇬🇧 English

### Welcome Contributors!

AI4S-agent-tools is an open-source project maintained by the DeepModeling community, aimed at building an intelligent agent tool library for scientific research. We welcome all forms of contributions!

### 📋 Before Contributing

1. **Fork the repository** - Click the Fork button in the top right
2. **Clone locally**
   ```bash
   git clone https://github.com/your-username/AI4S-agent-tools.git
   cd AI4S-agent-tools
   ```
3. **Install UV** (Python dependency manager)
   ```bash
   pip install uv
   ```
4. **Understand project structure**
   ```
   AI4S-agent-tools/
   ├── servers/           # All MCP servers
   │   ├── _example/      # Template example
   │   └── your_tool/     # Your new tool
   ├── scripts/           # Utility scripts
   ├── config/            # Configuration files
   ├── showcase/          # Showcase page (auto-generated)
   └── TOOLS.json         # Tool registry (auto-generated)
   ```

### 🚀 Quick Start: Adding a New Tool

#### Method 1: Using Template (Recommended)

1. **Copy the example server**
   ```bash
   cp -r servers/_example servers/your_tool_name
   ```

2. **Modify server.py**
   ```python
   from servers.server_utils import mcp_server, setup_server
   from mcp.server.fastmcp import FastMCP

   @mcp_server("YourToolName", "Clear description of your tool", author="@your-github", category="Category")
   def create_server(host="0.0.0.0", port=50001):
       mcp = FastMCP("your_tool", host=host, port=port)
       
       @mcp.tool()
       def your_function(param: str) -> dict:
           """Function documentation"""
           try:
               # Implement your scientific logic here
               return {"result": "value"}
           except Exception as e:
               return {"error": f"Failed: {str(e)}"}
       
       return mcp

   if __name__ == "__main__":
       setup_server().run()
   ```

3. **Update dependencies** (`pyproject.toml`)
   ```toml
   [project]
   name = "your-tool-name"
   version = "0.1.0"
   description = "Tool description"
   requires-python = ">=3.8"
   dependencies = [
       "fastmcp>=0.5.0",
       # Add your other dependencies
   ]
   ```

4. **Add documentation** (`README.md`)
   ```markdown
   # Your Tool Name

   ## Features
   Detailed description of what your tool does...

   ## Usage Example
   ```python
   # Show how to use your tool
   ```

   ## Installation and Running
   ```bash
   cd servers/your_tool_name
   uv sync
   python server.py --port 50001
   ```
   ```

5. **Choose the right category**
   
   Use one of these categories in the `@mcp_server` decorator:
   - `"Materials Science"` - Materials-related tools
   - `"Chemistry"` - Chemical calculations and analysis
   - `"Biology"` - Biological systems analysis
   - `"Physics"` - Physical simulations and calculations
   - `"Research Tools"` - Literature search and knowledge management
   - `"Simulation"` - Molecular dynamics and modeling
   - `"Data & Analysis"` - Data processing and visualization
   - `"Machine Learning"` - AI/ML models
   - `"General Tools"` - General purpose tools

6. **Update tool registry**
   ```bash
   python scripts/generate_tools_json.py
   ```

### 📝 Code Standards

1. **Follow Existing Patterns**
   - Use `@mcp_server` decorator to register servers
   - Use `@mcp.tool()` decorator to define tool functions
   - Always include error handling
   - Specify correct category in `@mcp_server` decorator

2. **Naming Conventions**
   - Server directories: lowercase, underscore-separated
   - Python files: follow PEP 8
   - Tool functions: clear verb phrases describing functionality

3. **Documentation Requirements**
   - Each tool must have a README.md
   - Functions must have docstrings
   - Complex features need usage examples

4. **Testing Requirements**
   - Manually test all functionality
   - Provide test commands and expected results
   - Ensure server starts properly

### 🔧 Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and test**
   ```bash
   # Install dependencies
   cd servers/your_tool_name
   uv sync
   
   # Run server
   python server.py --port 50001 --log-level DEBUG
   
   # Check logs
   tail -f ~/.your_tool_name/*.log
   ```

3. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new tool - brief description"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### ⚠️ Important Notes

- **Port allocation**: Check TOOLS.json to avoid port conflicts
- **Dependency management**: Use UV, don't modify uv.lock directly
- **Security**: Never commit keys or sensitive information
- **Compatibility**: Ensure Python >= 3.8
- **Showcase page**: Your tool will automatically appear on the [project showcase](https://lhhhappy.github.io/AI4S-agent-tools/)

### 🎯 PR Checklist

Before submitting a PR, please confirm:

- [ ] Code runs properly
- [ ] Added necessary documentation
- [ ] Ran `generate_tools_json.py`
- [ ] Tested main functionality
- [ ] Code style is consistent
- [ ] No hardcoded paths or keys

### 💡 Tips for Success

1. **Start small** - Begin with a simple tool and expand
2. **Ask questions** - Open an issue if you need help
3. **Review examples** - Study existing servers for patterns
4. **Test thoroughly** - Your future self will thank you
5. **Document clearly** - Help others understand your work

### 🤝 Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check CLAUDE.md for AI assistant guidance

### 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## 🌟 Thank You!

Every contribution makes AI4S-agent-tools better. Whether it's adding a new tool, fixing bugs, improving documentation, or suggesting ideas - we appreciate your help in advancing scientific computing!

Happy coding! 🚀