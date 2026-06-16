import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state import init_session_state, render_sidebar_info


st.set_page_config(
    page_title="高光谱遥感影像分类系统",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
render_sidebar_info()

st.title("🛰️ 高光谱遥感影像分类与地物识别系统")
st.markdown("---")

st.markdown("""
### 欢迎使用高光谱影像分类系统

本系统支持从数据导入到分类出图的完整处理流程，请通过左侧导航栏选择功能模块：

| 模块 | 功能说明 |
|------|---------|
| 📥 数据导入 | 加载ENVI格式高光谱数据，预览影像和光谱曲线 |
| ⚙️ 数据预处理 | 噪声波段剔除、MNF/PCA降维、光谱平滑 |
| 🔬 特征提取 | 光谱特征、空间特征、融合特征提取 |
| 📝 样本管理 | 标签导入、ROI选取、SMOTE均衡、训练集划分 |
| 🤖 模型训练 | SVM/随机森林/1D-CNN/3D-CNN/半监督学习 |
| 📊 精度评估 | OA/AA/Kappa/F1指标、混淆矩阵 |
| 🖼️ 分类可视化 | 分类图、叠加显示、单类提取、波段组合 |
| 📦 批量处理 | 多景影像批量分类、分块推理 |

### 处理流程

1. **数据导入** → 上传 `.hdr` + 数据文件
2. **数据预处理** → 可选的噪声剔除/降维/平滑
3. **特征提取** → 选择光谱/空间/融合特征
4. **样本管理** → 导入标签或交互选取ROI，划分训练/测试集
5. **模型训练** → 选择分类器，训练并执行全图分类
6. **精度评估** → 查看分类精度指标和混淆矩阵
7. **分类可视化** → 分类图展示、下载结果
8. **批量处理** → 对多景影像批量分类
""")

st.markdown("---")
st.markdown("🚀 高光谱遥感影像分类与地物识别系统 v1.0")
