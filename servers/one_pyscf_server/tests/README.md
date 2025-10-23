# PySCF MCP Server 测试文档

本目录包含了 PySCF MCP Server 的完整测试套件，用于验证服务器功能的正确性和稳定性。

## 目录结构

```
tests/
├── README.md              # 本文档
├── run_tests.py           # 测试运行脚本
├── test_server.py         # 主要测试文件
├── pyscf_codes/           # 测试用的PySCF代码
│   ├── simple_dft.py      # 正常的DFT计算代码
│   └── invalid_code.py    # 包含错误的代码（用于测试错误处理）
└── test_xyz/              # 测试用的分子结构文件
    ├── H2O.xyz            # 水分子
    ├── test_molecule.xyz  # 测试分子（苯环）
    └── invalid.xyz        # 格式错误的XYZ文件
```

## 测试覆盖范围

### 正常功能测试
1. **基本PySCF计算** - 使用水分子进行DFT计算
2. **不同分子计算** - 使用不同的分子结构文件
3. **不使用参数的代码** - 测试不依赖XYZ文件的代码

### 异常情况测试
1. **文件不存在** - XYZ文件路径无效
2. **语法错误** - PySCF代码包含语法错误
3. **运行时错误** - 代码执行时出现异常
4. **导入错误** - 导入不存在的模块
5. **无效XYZ格式** - XYZ文件格式不正确
6. **空代码** - 提供空的PySCF代码

## 运行测试

### 方式1: 使用测试运行脚本（推荐）

```bash
cd /Users/xhxu/Documents/AI4S-agent-tools/servers/one_pyscf_server/tests
python run_tests.py
```

这个脚本会：
- 检查所需依赖是否安装
- 验证测试文件是否存在
- 使用pytest和直接运行两种方式执行测试
- 提供详细的测试报告

### 方式2: 使用pytest

```bash
cd /Users/xhxu/Documents/AI4S-agent-tools/servers/one_pyscf_server
python -m pytest tests/test_server.py -v
```

### 方式3: 直接运行测试文件

```bash
cd /Users/xhxu/Documents/AI4S-agent-tools/servers/one_pyscf_server
python tests/test_server.py
```

## 依赖要求

运行测试需要以下Python包：

```
pytest>=6.0.0
pyscf>=2.0.0
asyncio (Python标准库)
pathlib (Python标准库)
```

安装依赖：
```bash
pip install pytest pyscf
```

## 测试说明

### TestPySCFServer 类

这是主要的测试类，包含以下测试方法：

- `test_run_pyscf_code_success()` - 测试正常的PySCF计算
- `test_run_pyscf_code_with_different_molecule()` - 测试不同分子的计算
- `test_run_pyscf_code_file_not_found()` - 测试文件不存在的情况
- `test_run_pyscf_code_syntax_error()` - 测试语法错误
- `test_run_pyscf_code_runtime_error()` - 测试运行时错误
- `test_run_pyscf_code_import_error()` - 测试导入错误
- `test_run_pyscf_code_invalid_xyz_format()` - 测试无效XYZ格式
- `test_run_pyscf_code_empty_code()` - 测试空代码
- `test_run_pyscf_code_no_args()` - 测试不使用参数的代码

### 测试数据

#### PySCF代码文件
- `simple_dft.py`: 使用PySCF进行DFT计算的标准代码
- `invalid_code.py`: 包含导入错误的代码，用于测试错误处理

#### XYZ文件
- `H2O.xyz`: 水分子的几何结构
- `test_molecule.xyz`: 苯环分子的几何结构
- `invalid.xyz`: 格式错误的文件，用于测试错误处理

## 预期结果

正常情况下，所有测试都应该通过。测试结果会显示：

- ✅ 成功的测试会显示绿色勾号和简要信息
- ❌ 失败的测试会显示红色叉号和错误信息
- 异常情况测试验证服务器能正确处理各种错误情况

## 故障排除

### 常见问题

1. **ImportError: No module named 'pyscf'**
   - 解决方案: `pip install pyscf`

2. **ImportError: No module named 'pytest'**
   - 解决方案: `pip install pytest`

3. **FileNotFoundError: 测试文件不存在**
   - 检查测试文件路径是否正确
   - 确保在正确的目录下运行测试

4. **服务器导入错误**
   - 确保 `server.py` 文件存在且可以正常导入
   - 检查Python路径设置

### 调试建议

1. 单独运行特定测试：
   ```bash
   python -m pytest tests/test_server.py::TestPySCFServer::test_run_pyscf_code_success -v
   ```

2. 查看详细输出：
   ```bash
   python -m pytest tests/test_server.py -v -s
   ```

3. 在测试代码中添加调试信息：
   ```python
   print(f"Debug: {result}")
   ```

## 贡献

如果需要添加新的测试用例：

1. 在 `TestPySCFServer` 类中添加新的测试方法
2. 方法名以 `test_` 开头
3. 使用 `@pytest.mark.asyncio` 装饰器（如果是异步测试）
4. 添加适当的断言验证结果
5. 更新 `test_sync_wrapper()` 函数以包含新测试
6. 更新本文档

## 许可证

本测试套件遵循与主项目相同的许可证。