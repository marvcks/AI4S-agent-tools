# AI4S Agent Tools 工具展示方案

本目录包含了一套完整的工具展示和管理方案，为 AI4S Agent Tools 项目提供优雅的展示、监控和管理功能。

## 🚀 功能概览

### 1. 工具展示网站生成器 (`generate_tools_showcase.py`)
- **功能**：生成一个美观的静态网站来展示所有工具
- **特性**：
  - 响应式设计，支持移动端访问
  - 工具搜索和分类过滤
  - 交互式统计图表
  - 工具详情模态框
  - Mermaid 依赖关系图
  - 自动生成统计报告

**使用方法**：
```bash
python scripts/generate_tools_showcase.py
# 生成的网站位于 showcase/index.html
```

### 2. 实时监控仪表板 (`tools_dashboard.py`)
- **功能**：提供工具运行状态的实时监控和管理
- **特性**：
  - WebSocket 实时数据更新
  - CPU 和内存使用率监控
  - 工具启动/停止控制
  - 实时性能图表
  - 运行状态统计

**使用方法**：
```bash
python scripts/tools_dashboard.py --host 127.0.0.1 --port 8080
# 访问 http://127.0.0.1:8080 查看仪表板
```

### 3. 依赖关系分析器 (`generate_dependency_graph.py`)
- **功能**：分析工具之间的依赖关系并生成可视化图表
- **特性**：
  - 自动分析 pyproject.toml 文件
  - 生成依赖关系网络图
  - 识别共享依赖
  - 输出 Mermaid 格式图表
  - 生成详细的依赖报告

**使用方法**：
```bash
python scripts/generate_dependency_graph.py
# 输出位于 showcase/dependencies/
```

### 4. 命令行管理工具 (`ai4s_tools_cli.py`)
- **功能**：提供统一的命令行界面来管理所有工具
- **特性**：
  - 列出和搜索工具
  - 显示工具详情
  - 启动和测试工具
  - 生成 MCP 客户端配置
  - 显示统计信息

**使用方法**：
```bash
# 列出所有工具
python scripts/ai4s_tools_cli.py list

# 搜索工具
python scripts/ai4s_tools_cli.py search "chemistry"

# 显示工具详情
python scripts/ai4s_tools_cli.py show DPACalculatorServer

# 启动工具
python scripts/ai4s_tools_cli.py start pubchem --port 50001

# 生成配置
python scripts/ai4s_tools_cli.py config -o mcp_config.json

# 显示统计
python scripts/ai4s_tools_cli.py stats

# 更新工具列表
python scripts/ai4s_tools_cli.py update
```

## 📦 依赖要求

这些脚本需要以下 Python 包：

```bash
# 基础依赖
pip install fastapi uvicorn websockets psutil

# 可视化依赖
pip install matplotlib networkx

# CLI 依赖
pip install rich requests

# Web 依赖（通过 CDN 加载，无需安装）
# - Tailwind CSS
# - Font Awesome
# - Chart.js
# - Mermaid.js
```

## 🎯 使用场景

### 场景 1：项目展示
1. 运行 `generate_tools_showcase.py` 生成展示网站
2. 将 `showcase/` 目录部署到 GitHub Pages 或其他静态网站托管服务
3. 分享链接给用户和贡献者

### 场景 2：开发调试
1. 使用 `ai4s_tools_cli.py` 快速启动需要的工具
2. 运行 `tools_dashboard.py` 监控工具性能
3. 使用仪表板实时查看日志和资源使用情况

### 场景 3：依赖管理
1. 运行 `generate_dependency_graph.py` 分析依赖
2. 查看生成的报告了解共享依赖
3. 优化依赖结构，减少冗余

### 场景 4：文档生成
1. 所有脚本都会自动生成相应的文档
2. `showcase/README.md` 包含工具概览
3. `showcase/statistics.json` 提供详细统计数据

## 🔧 扩展建议

### 1. 集成 CI/CD
- 在 GitHub Actions 中自动运行展示生成器
- 自动部署到 GitHub Pages
- 定期更新统计数据

### 2. 添加更多功能
- 工具使用教程生成
- API 文档自动生成
- 性能基准测试
- 用户反馈收集

### 3. 改进监控
- 添加日志聚合
- 实现告警功能
- 历史数据存储
- 性能趋势分析

## 📝 注意事项

1. **端口管理**：确保为每个工具分配唯一的端口
2. **依赖更新**：定期运行 `update` 命令更新工具列表
3. **资源监控**：使用仪表板避免资源过度使用
4. **安全考虑**：仅在受信任的环境中运行监控仪表板

## 🤝 贡献

欢迎贡献新的展示功能或改进现有工具！请确保：
- 遵循项目的代码风格
- 添加适当的文档
- 测试所有功能
- 提交 PR 前运行 `ai4s_tools_cli.py update`