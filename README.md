# kNN Weakness Project (A3)

本仓库为课程 A3 项目提交材料，主题为 **kNN 算法弱点分析与反例构造**。

## 📁 项目结构
- `report.md`：实验报告（含图表）
- `llm_dialogue.md`：大模型对话全过程记录
- `reflection.md` / `thinking_log.md`：思想日志
- `iterations.md`：v1 / v2 / v3 版本迭代说明
- `results/`：实验结果与可视化图片
- `notebooks/run_knn_demo.py`：实验代码

## 🎯 项目目标
通过构造特定数据分布，验证并分析 kNN 在以下场景下的弱点：
- 噪声敏感性
- 维度灾难
- 不同密度聚类
- 计算效率问题

## 🤖 人机协作说明
本项目在实验设计阶段参考了大模型的理论建议，但所有参数选择、
实验验证与结论均由本人独立完成并验证。

## 📌 提交说明

本项目为 A3 课程作业，包含以下四类材料：

1. **实验报告**：report.md  
2. **大模型对话记录**：llm_dialogue.md  
3. **思想日志**：reflection.md  
4. **版本迭代说明**：iterations.md  

所有实验结果图像位于 `results/` 目录中，  
完整实验代码位于 `notebooks/run_knn_demo.py`。

本仓库完整记录了从任务理解、实验设计、模型验证到反思总结的全过程。
## ▶️ 复现方式（Windows / PowerShell）

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python notebooks/run_knn_demo.py

### 环境要求
- Python 3.8+
- Git（可选）

### 安装步骤

#### 方法1：使用虚拟环境（推荐）
```powershell
# 克隆项目
git clone https://github.com/sumikko-Charlotte/knn_weakness_project.git
cd knn_weakness_project

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python notebooks/run_knn_demo.py