import ast
import loguru
from typing import Set, List, Dict, Any

# --- 配置区：定义不允许的操作 ---

# 1. 禁止导入的模块
# 这些模块提供了与操作系统、文件系统、网络和子进程交互的能力。
FORBIDDEN_IMPORTS: Set[str] = {
    'os',
    'subprocess',
    'shutil',
    'requests',
    'urllib',
    'socket',
    'sys',
    'platform',
    'importlib',
    'ctypes',
    'multiprocessing',
    'threading',
}

# 2. 禁止调用的内置函数
# 这些函数可以执行代码、读写文件或动态获取属性，非常危险。
FORBIDDEN_BUILTINS: Set[str] = {
    'eval',
    'exec',
    '__import__',
    'getattr',  # 限制getattr，防止 getattr(os, 'system') 这种绕过
    'setattr',
    'delattr',
}

# 3. 禁止的属性/方法调用
# 例如, "os.system"
FORBIDDEN_CALLS: Set[str] = {
    'system',
    'popen',
    'spawn',
    'fork',
    'kill',
    'execv',
    'execve',
    'listdir',
    'walk',
    'remove',
    'rmdir',
    'unlink',
    'putenv',
    'environ',
}

# --- AST分析器 ---

class CodeVisitor(ast.NodeVisitor):
    """
    一个AST节点访问器，用于遍历代码树并找出不安全的操作。
    """
    def __init__(self):
        self.violations: List[Dict[str, Any]] = []

    def visit_Import(self, node: ast.Import):
        """处理 'import os' 这种形式的导入"""
        for alias in node.names:
            if alias.name in FORBIDDEN_IMPORTS:
                self.violations.append({
                    "type": "Forbidden Import",
                    "module": alias.name,
                    "line": node.lineno,
                    "reason": f"Importing the module '{alias.name}' is not allowed."
                })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """处理 'from os import system' 这种形式的导入"""
        if node.module and node.module in FORBIDDEN_IMPORTS:
            self.violations.append({
                "type": "Forbidden Import From",
                "module": node.module,
                "line": node.lineno,
                "reason": f"Importing from the module '{node.module}' is not allowed."
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """处理函数调用，例如 open(...) 或 os.system(...)"""
        # 检查内置函数调用，如 eval("...")
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
            self.violations.append({
                "type": "Forbidden Builtin Call",
                "function": node.func.id,
                "line": node.lineno,
                "reason": f"Calling the built-in function '{node.func.id}' is not allowed."
            })
        
        # 检查属性方法调用，如 os.system("...")
        if isinstance(node.func, ast.Attribute):
            # 检查方法名是否在禁止列表中
            if node.func.attr in FORBIDDEN_CALLS:
                 self.violations.append({
                    "type": "Forbidden Method Call",
                    "method": node.func.attr,
                    "line": node.lineno,
                    "reason": f"Calling the method '{node.func.attr}' is not allowed."
                })

        self.generic_visit(node)

# --- 主检查函数 ---

def is_safe(code: str, logger) -> bool:
    """
    对给定的Python代码字符串进行静态分析，检查是否存在不安全的操作。

    重要提示: 这个函数是一个初步的、尽力而为的安全检查，
    它不能替代一个真正的沙箱环境。它的目的是在代码进入沙箱前
    过滤掉明显不安全的脚本。

    Args:
        code (str): 由AI生成的Python代码字符串。
        logger (logging.Logger): 用于记录日志的日志器实例。

    Returns:
        bool: 如果代码通过所有检查则返回 True，否则返回 False。
    """
    logger.info("Starting static analysis on generated code...")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Code failed to parse with syntax error: {e}")
        return False

    visitor = CodeVisitor()
    visitor.visit(tree)

    if visitor.violations:
        for violation in visitor.violations:
            logger.warning(
                f"Security violation detected at line {violation['line']}: "
                f"{violation['type']} -> {violation['reason']}"
            )
        return False

    logger.info("Static analysis passed. No obvious violations found.")
    return True

if __name__ == "__main__":
    from loguru import logger
    test_code = "a = 10\nprint(a)\nopen('test.txt', 'w')"
    safe = is_safe(test_code, logger)
    print(f"Is the code safe? {safe}")
