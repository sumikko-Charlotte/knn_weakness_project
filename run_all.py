#!/usr/bin/env python3
"""
kNN弱点分析项目 - 一键运行脚本
运行所有实验并生成图表
"""
import os
import subprocess
import sys

def check_environment():
    """检查Python环境和依赖"""
    print("=" * 60)
    print("检查Python环境...")
    
    # 检查Python版本
    import platform
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    
    # 检查必要库
    required_libs = ['numpy', 'pandas', 'matplotlib', 'sklearn']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} 已安装")
        except ImportError:
            missing_libs.append(lib)
            print(f"❌ {lib} 未安装")
    
    if missing_libs:
        print(f"\n缺少依赖: {missing_libs}")
        print("请运行: pip install " + " ".join(missing_libs))
        return False
    
    print("✅ 环境检查通过")
    return True

def run_notebooks():
    """运行所有Jupyter notebook"""
    notebooks = [
        "notebooks/run_knn_demo.py",  # 你的主脚本
        # 可以添加更多脚本
    ]
    
    print("\n" + "=" * 60)
    print("开始运行实验...")
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            print(f"运行: {notebook}")
            try:
                # 执行Python脚本
                result = subprocess.run([sys.executable, notebook], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {notebook} 运行成功")
                else:
                    print(f"❌ {notebook} 运行失败")
                    print("错误信息:", result.stderr)
            except Exception as e:
                print(f"❌ 运行{notebook}时出错: {e}")
        else:
            print(f"⚠️ 文件不存在: {notebook}")
    
    return True

def check_results():
    """检查生成的图片"""
    print("\n" + "=" * 60)
    print("检查生成结果...")
    
    expected_files = [
        "results/decision_boundary_case1.png",
        "results/dim_curse.png", 
        "results/varied_density.png"
    ]
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建目录: {results_dir}")
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✅ {file} 已生成 ({size:.1f} KB)")
        else:
            print(f"❌ {file} 未生成")
    
    # 列出所有结果文件
    print(f"\nresults/目录内容:")
    for file in os.listdir(results_dir):
        filepath = os.path.join(results_dir, file)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"  - {file} ({size:.1f} KB)")

def main():
    """主函数"""
    print("=" * 60)
    print("kNN弱点分析项目 - 自动化运行脚本")
    print("=" * 60)
    
    # 1. 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 2. 运行实验
    run_notebooks()
    
    # 3. 检查结果
    check_results()
    
    print("\n" + "=" * 60)
    print("✅ 所有任务完成！")
    print("生成的图片保存在 results/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()