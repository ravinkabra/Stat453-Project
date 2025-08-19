from pathlib import Path
import os

# --- 配置 ---
# 将 'src' 改为你想开始搜索的目录。
# 如果想从当前项目根目录开始，就用 '.'
start_directory = "src"
# --- 结束配置 ---

# 获取绝对路径
dir_path = Path(start_directory).resolve()

if not dir_path.is_dir():
    print(f"错误: 目录 '{start_directory}' 不存在。")
else:
    # 使用 os.walk 遍历所有子目录
    for root, dirs, files in os.walk(dir_path):
        # 检查 __init__.py 是否需要创建
        if "__pycache__" in root:
            continue  # 跳过 __pycache__ 目录

        init_py_path = Path(root) / "__init__.py"
        if not init_py_path.exists():
            print(f"正在创建: {init_py_path}")
            init_py_path.touch()  # .touch() 会创建一个空文件

    print("\n完成！")
