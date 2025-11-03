#!/usr/bin/env python3
"""
故意包含错误的PySCF代码，用于测试错误处理
"""

import sys
xyz_path = sys.argv[1]

# 故意的导入错误
from nonexistent_module import something

# 这行代码不会被执行
print("This should not be reached")