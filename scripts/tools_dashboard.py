#!/usr/bin/env python3
"""
AI4S Agent Tools 实时监控仪表板
提供工具运行状态监控、使用统计和管理功能
"""
import json
import asyncio
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI4S Tools Dashboard")

# 全局状态存储
TOOLS_STATUS = {}
TOOLS_METRICS = {}
ACTIVE_CONNECTIONS = []

class ToolManager:
    """工具管理器"""
    
    def __init__(self, tools_json_path: Path):
        self.tools_json_path = tools_json_path
        self.load_tools()
        self.processes = {}
        
    def load_tools(self):
        """加载工具配置"""
        with open(self.tools_json_path, 'r', encoding='utf-8') as f:
            self.tools_data = json.load(f)
    
    async def start_tool(self, tool_name: str, port: int) -> bool:
        """启动工具服务器"""
        tool = next((t for t in self.tools_data['tools'] if t['name'] == tool_name), None)
        if not tool:
            return False
        
        try:
            # 构建启动命令
            cmd = tool['start_command'].replace('<PORT>', str(port))
            
            # 启动进程
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            self.processes[tool_name] = {
                'process': process,
                'port': port,
                'started_at': datetime.now().isoformat()
            }
            
            # 更新状态
            TOOLS_STATUS[tool_name] = {
                'status': 'running',
                'port': port,
                'pid': process.pid,
                'started_at': datetime.now().isoformat()
            }
            
            return True
        except Exception as e:
            logger.error(f"启动工具 {tool_name} 失败: {e}")
            return False
    
    async def stop_tool(self, tool_name: str) -> bool:
        """停止工具服务器"""
        if tool_name not in self.processes:
            return False
        
        try:
            process_info = self.processes[tool_name]
            process = process_info['process']
            
            # 终止进程
            process.terminate()
            await process.wait()
            
            # 清理状态
            del self.processes[tool_name]
            TOOLS_STATUS[tool_name] = {'status': 'stopped'}
            
            return True
        except Exception as e:
            logger.error(f"停止工具 {tool_name} 失败: {e}")
            return False
    
    async def get_tool_metrics(self, tool_name: str) -> Dict[str, Any]:
        """获取工具性能指标"""
        if tool_name not in self.processes:
            return {}
        
        try:
            process_info = self.processes[tool_name]
            pid = process_info['process'].pid
            
            # 使用 psutil 获取进程信息
            proc = psutil.Process(pid)
            
            metrics = {
                'cpu_percent': proc.cpu_percent(interval=0.1),
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'threads': proc.num_threads(),
                'status': proc.status(),
                'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
            }
            
            return metrics
        except Exception as e:
            logger.error(f"获取工具 {tool_name} 指标失败: {e}")
            return {}

# 初始化工具管理器
tool_manager = None

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global tool_manager
    root_dir = Path(__file__).parent.parent
    tools_json_path = root_dir / 'TOOLS.json'
    tool_manager = ToolManager(tools_json_path)
    
    # 初始化所有工具状态
    for tool in tool_manager.tools_data['tools']:
        TOOLS_STATUS[tool['name']] = {'status': 'stopped'}
        TOOLS_METRICS[tool['name']] = {}

@app.get("/")
async def dashboard():
    """仪表板主页"""
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/api/tools")
async def get_tools():
    """获取所有工具信息"""
    return {
        'tools': tool_manager.tools_data['tools'],
        'status': TOOLS_STATUS,
        'metrics': TOOLS_METRICS
    }

@app.post("/api/tools/{tool_name}/start")
async def start_tool(tool_name: str, port: int):
    """启动指定工具"""
    success = await tool_manager.start_tool(tool_name, port)
    if success:
        return {"message": f"工具 {tool_name} 已启动在端口 {port}"}
    else:
        raise HTTPException(status_code=500, detail="启动工具失败")

@app.post("/api/tools/{tool_name}/stop")
async def stop_tool(tool_name: str):
    """停止指定工具"""
    success = await tool_manager.stop_tool(tool_name)
    if success:
        return {"message": f"工具 {tool_name} 已停止"}
    else:
        raise HTTPException(status_code=500, detail="停止工具失败")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点用于实时更新"""
    await websocket.accept()
    ACTIVE_CONNECTIONS.append(websocket)
    
    try:
        while True:
            # 定期发送状态更新
            await asyncio.sleep(2)
            
            # 更新所有运行中工具的指标
            for tool_name in list(tool_manager.processes.keys()):
                metrics = await tool_manager.get_tool_metrics(tool_name)
                TOOLS_METRICS[tool_name] = metrics
            
            # 发送更新
            await websocket.send_json({
                'type': 'update',
                'status': TOOLS_STATUS,
                'metrics': TOOLS_METRICS,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        ACTIVE_CONNECTIONS.remove(websocket)

# 仪表板 HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI4S Tools 监控仪表板</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-running { color: #10b981; }
        .status-stopped { color: #ef4444; }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
</head>
<body class="bg-gray-100">
    <nav class="bg-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <h1 class="text-2xl font-bold text-gray-800">
                    <i class="fas fa-tachometer-alt mr-2"></i>AI4S Tools 监控仪表板
                </h1>
                <div class="text-sm text-gray-600">
                    <i class="fas fa-circle text-green-500 mr-1"></i>
                    连接状态: <span id="connectionStatus">已连接</span>
                </div>
            </div>
        </div>
    </nav>

    <div class="container mx-auto px-6 py-8">
        <!-- 统计卡片 -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="metric-card rounded-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white opacity-75">总工具数</p>
                        <p class="text-3xl font-bold" id="totalTools">0</p>
                    </div>
                    <i class="fas fa-tools text-4xl opacity-50"></i>
                </div>
            </div>
            
            <div class="metric-card rounded-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white opacity-75">运行中</p>
                        <p class="text-3xl font-bold" id="runningTools">0</p>
                    </div>
                    <i class="fas fa-play-circle text-4xl opacity-50"></i>
                </div>
            </div>
            
            <div class="metric-card rounded-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white opacity-75">CPU 使用率</p>
                        <p class="text-3xl font-bold" id="totalCpu">0%</p>
                    </div>
                    <i class="fas fa-microchip text-4xl opacity-50"></i>
                </div>
            </div>
            
            <div class="metric-card rounded-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white opacity-75">内存使用</p>
                        <p class="text-3xl font-bold" id="totalMemory">0MB</p>
                    </div>
                    <i class="fas fa-memory text-4xl opacity-50"></i>
                </div>
            </div>
        </div>

        <!-- 工具列表 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-xl font-bold mb-4">工具状态</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">工具名称</th>
                            <th class="text-left py-2">状态</th>
                            <th class="text-left py-2">端口</th>
                            <th class="text-left py-2">CPU</th>
                            <th class="text-left py-2">内存</th>
                            <th class="text-left py-2">操作</th>
                        </tr>
                    </thead>
                    <tbody id="toolsTableBody">
                        <!-- 动态生成 -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 实时图表 -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-bold mb-4">CPU 使用趋势</h3>
                <canvas id="cpuChart"></canvas>
            </div>
            
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-bold mb-4">内存使用趋势</h3>
                <canvas id="memoryChart"></canvas>
            </div>
        </div>
    </div>

    <!-- 启动工具模态框 -->
    <div id="startModal" class="fixed inset-0 z-50 hidden">
        <div class="flex items-center justify-center min-h-screen px-4">
            <div class="fixed inset-0 bg-gray-500 bg-opacity-75"></div>
            <div class="relative bg-white rounded-lg max-w-md w-full p-6">
                <h3 class="text-lg font-bold mb-4">启动工具</h3>
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-2">端口号</label>
                    <input type="number" id="portInput" 
                           class="w-full px-3 py-2 border border-gray-300 rounded-lg"
                           placeholder="例如: 50001">
                </div>
                <div class="flex justify-end space-x-2">
                    <button onclick="closeStartModal()" 
                            class="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400">
                        取消
                    </button>
                    <button onclick="confirmStart()" 
                            class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                        启动
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let toolsData = {};
        let selectedTool = null;
        let cpuChart = null;
        let memoryChart = null;
        let chartData = {
            labels: [],
            cpu: {},
            memory: {}
        };

        // 初始化
        async function init() {
            // 连接 WebSocket
            connectWebSocket();
            
            // 加载初始数据
            await loadTools();
            
            // 初始化图表
            initCharts();
        }

        // WebSocket 连接
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('connectionStatus').textContent = '已连接';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'update') {
                    updateDashboard(data);
                }
            };
            
            ws.onclose = () => {
                document.getElementById('connectionStatus').textContent = '已断开';
                setTimeout(connectWebSocket, 5000); // 5秒后重连
            };
        }

        // 加载工具数据
        async function loadTools() {
            const response = await fetch('/api/tools');
            const data = await response.json();
            toolsData = data;
            renderToolsTable();
            updateStats();
        }

        // 渲染工具表格
        function renderToolsTable() {
            const tbody = document.getElementById('toolsTableBody');
            tbody.innerHTML = toolsData.tools.map(tool => {
                const status = toolsData.status[tool.name] || {};
                const metrics = toolsData.metrics[tool.name] || {};
                const isRunning = status.status === 'running';
                
                return `
                    <tr class="border-b">
                        <td class="py-2">${tool.name}</td>
                        <td class="py-2">
                            <span class="${isRunning ? 'status-running' : 'status-stopped'}">
                                <i class="fas fa-circle mr-1"></i>
                                ${isRunning ? '运行中' : '已停止'}
                            </span>
                        </td>
                        <td class="py-2">${status.port || '-'}</td>
                        <td class="py-2">${metrics.cpu_percent ? metrics.cpu_percent.toFixed(1) + '%' : '-'}</td>
                        <td class="py-2">${metrics.memory_mb ? metrics.memory_mb.toFixed(1) + 'MB' : '-'}</td>
                        <td class="py-2">
                            ${isRunning ? 
                                `<button onclick="stopTool('${tool.name}')" 
                                         class="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600">
                                    停止
                                </button>` :
                                `<button onclick="startTool('${tool.name}')" 
                                         class="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600">
                                    启动
                                </button>`
                            }
                        </td>
                    </tr>
                `;
            }).join('');
        }

        // 更新统计数据
        function updateStats() {
            const runningCount = Object.values(toolsData.status).filter(s => s.status === 'running').length;
            const totalCpu = Object.values(toolsData.metrics).reduce((sum, m) => sum + (m.cpu_percent || 0), 0);
            const totalMemory = Object.values(toolsData.metrics).reduce((sum, m) => sum + (m.memory_mb || 0), 0);
            
            document.getElementById('totalTools').textContent = toolsData.tools.length;
            document.getElementById('runningTools').textContent = runningCount;
            document.getElementById('totalCpu').textContent = totalCpu.toFixed(1) + '%';
            document.getElementById('totalMemory').textContent = totalMemory.toFixed(1) + 'MB';
        }

        // 更新仪表板
        function updateDashboard(data) {
            toolsData.status = data.status;
            toolsData.metrics = data.metrics;
            
            renderToolsTable();
            updateStats();
            updateCharts(data);
        }

        // 初始化图表
        function initCharts() {
            const cpuCtx = document.getElementById('cpuChart').getContext('2d');
            cpuChart = new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            const memoryCtx = document.getElementById('memoryChart').getContext('2d');
            memoryChart = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // 更新图表
        function updateCharts(data) {
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            
            // 限制数据点数量
            if (chartData.labels.length > 20) {
                chartData.labels.shift();
                Object.keys(chartData.cpu).forEach(tool => {
                    chartData.cpu[tool].shift();
                    chartData.memory[tool].shift();
                });
            }
            
            chartData.labels.push(timestamp);
            
            // 更新每个工具的数据
            Object.entries(data.metrics).forEach(([tool, metrics]) => {
                if (metrics.cpu_percent !== undefined) {
                    if (!chartData.cpu[tool]) chartData.cpu[tool] = [];
                    if (!chartData.memory[tool]) chartData.memory[tool] = [];
                    
                    chartData.cpu[tool].push(metrics.cpu_percent);
                    chartData.memory[tool].push(metrics.memory_mb);
                }
            });
            
            // 更新图表
            updateChart(cpuChart, chartData.cpu, 'CPU %');
            updateChart(memoryChart, chartData.memory, 'Memory MB');
        }

        // 更新单个图表
        function updateChart(chart, data, label) {
            const datasets = Object.entries(data)
                .filter(([tool, values]) => values.length > 0)
                .map(([tool, values], index) => ({
                    label: tool,
                    data: values,
                    borderColor: getColor(index),
                    backgroundColor: getColor(index, 0.1),
                    tension: 0.1
                }));
            
            chart.data.labels = chartData.labels;
            chart.data.datasets = datasets;
            chart.update();
        }

        // 获取颜色
        function getColor(index, alpha = 1) {
            const colors = [
                `rgba(59, 130, 246, ${alpha})`,
                `rgba(239, 68, 68, ${alpha})`,
                `rgba(34, 197, 94, ${alpha})`,
                `rgba(245, 158, 11, ${alpha})`,
                `rgba(139, 92, 246, ${alpha})`
            ];
            return colors[index % colors.length];
        }

        // 启动工具
        function startTool(toolName) {
            selectedTool = toolName;
            document.getElementById('startModal').classList.remove('hidden');
        }

        // 关闭启动模态框
        function closeStartModal() {
            document.getElementById('startModal').classList.add('hidden');
            selectedTool = null;
        }

        // 确认启动
        async function confirmStart() {
            const port = document.getElementById('portInput').value;
            if (!port) {
                alert('请输入端口号');
                return;
            }
            
            try {
                const response = await fetch(`/api/tools/${selectedTool}/start?port=${port}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    closeStartModal();
                    await loadTools();
                } else {
                    alert('启动失败');
                }
            } catch (error) {
                alert('启动失败: ' + error.message);
            }
        }

        // 停止工具
        async function stopTool(toolName) {
            if (!confirm(`确定要停止 ${toolName} 吗？`)) return;
            
            try {
                const response = await fetch(`/api/tools/${toolName}/stop`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    await loadTools();
                } else {
                    alert('停止失败');
                }
            } catch (error) {
                alert('停止失败: ' + error.message);
            }
        }

        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI4S Tools 监控仪表板')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    parser.add_argument('--port', type=int, default=8080, help='监听端口')
    
    args = parser.parse_args()
    
    print(f"🚀 启动 AI4S Tools 监控仪表板")
    print(f"📍 访问地址: http://{args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()