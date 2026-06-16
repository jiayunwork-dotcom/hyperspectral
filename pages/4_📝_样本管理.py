import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from state import init_session_state, render_sidebar_info
from src.sample_management import (
    ROIRegion, extract_labeled_samples, add_roi_to_samples,
    apply_smote, split_samples, get_sample_stats, create_empty_samples
)
from src.visualization import get_true_color, plot_class_distribution


init_session_state()
render_sidebar_info()

st.header("📝 训练样本管理")

if st.session_state.preprocessed_data is None:
    st.warning("⚠️ 请先完成数据导入和预处理")
    st.stop()

data = st.session_state.features if st.session_state.features is not None else st.session_state.preprocessed_data

tab1, tab2, tab3 = st.tabs(["📊 样本统计", "➕ 添加ROI样本", "⚖️ 样本均衡"])

with tab1:
    st.subheader("样本统计信息")
    if st.session_state.samples.n_samples == 0:
        st.info("ℹ️ 暂无训练样本，请先导入标签或添加ROI")
    else:
        stats = get_sample_stats(st.session_state.samples)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总样本数", stats['n_samples'])
        col2.metric("类别数", stats['n_classes'])
        col3.metric("是否均衡", "是" if stats['is_balanced'] else "否")
        col4.metric("均衡比例", f"{stats['balance_ratio']:.2f}")

        st.markdown("### 各类别样本数")
        dist_data = plot_class_distribution(stats['class_counts'], stats['class_names'])
        fig = px.bar(x=dist_data['labels'], y=dist_data['counts'], text=dist_data['counts'], title='样本类别分布')
        fig.update_traces(textposition='auto')
        fig.update_layout(xaxis_title='类别', yaxis_title='样本数')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 类别详情")
        df = pd.DataFrame({
            '类别': [stats['class_names'].get(c, f'Class {c}') for c in stats['class_counts'].keys()],
            '样本数': list(stats['class_counts'].values()),
            '比例': [f"{v*100:.2f}%" for v in stats['class_distribution'].values()]
        })
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("交互式ROI选取")
    preview_img = get_true_color(st.session_state.data, st.session_state.wavelengths)
    st.image(preview_img, caption="影像预览（用于参考ROI位置）", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ROI设置")
        roi_label = st.number_input("类别标签", 1, 100, 1, key='roi_label')
        roi_name = st.text_input("类别名称", f"类别 {roi_label}", key='roi_name')
        shape_type = st.selectbox("ROI形状", ["rectangle", "polygon"], format_func=lambda x: '矩形' if x == 'rectangle' else '多边形')

    with col2:
        st.markdown("### 坐标输入")
        H, W = st.session_state.data.shape[:2]
        if shape_type == 'rectangle':
            st.write("输入矩形对角点坐标 (行, 列)")
            y1 = st.number_input("左上角行", 0, H - 1, 0, key='y1')
            x1 = st.number_input("左上角列", 0, W - 1, 0, key='x1')
            y2 = st.number_input("右下角行", 0, H - 1, min(H-1, 50), key='y2')
            x2 = st.number_input("右下角列", 0, W - 1, min(W-1, 50), key='x2')
            coordinates = [(int(y1), int(x1)), (int(y2), int(x2))]
        else:
            st.write("输入多边形顶点坐标，每行一个顶点 (行, 列)")
            coords_text = st.text_area("顶点坐标", "10, 10\n10, 50\n30, 30", height=100, key='polygon_coords')
            try:
                coordinates = []
                for line in coords_text.strip().split('\n'):
                    if line.strip():
                        y, x = map(int, line.split(','))
                        coordinates.append((y, x))
            except:
                coordinates = []
                st.error("坐标格式错误")

    if st.button("➕ 添加ROI样本", type='primary'):
        if len(coordinates) < 2:
            st.warning("⚠️ 请输入有效的坐标")
            st.stop()
        roi = ROIRegion(label=int(roi_label), label_name=roi_name, coordinates=coordinates, shape_type=shape_type)
        try:
            new_samples = add_roi_to_samples(data, st.session_state.samples, roi)
            st.session_state.samples = new_samples
            st.success(f"✅ ROI添加成功！新增 {len(roi.pixels)} 个样本")
            st.rerun()
        except Exception as e:
            st.error(f"❌ ROI添加失败: {str(e)}")

    if len(st.session_state.samples.rois) > 0:
        st.markdown("### 已添加的ROI")
        roi_df = pd.DataFrame([{
            '序号': i + 1, '标签': roi.label, '名称': roi.label_name,
            '形状': '矩形' if roi.shape_type == 'rectangle' else '多边形', '像素数': len(roi.pixels)
        } for i, roi in enumerate(st.session_state.samples.rois)])
        st.dataframe(roi_df, use_container_width=True)
        if st.button("🗑️ 清空所有ROI"):
            st.session_state.samples = create_empty_samples()
            st.rerun()

with tab3:
    st.subheader("样本均衡处理")
    if st.session_state.samples.n_samples == 0:
        st.info("ℹ️ 暂无训练样本")
    elif st.session_state.samples.is_balanced:
        st.success("✅ 样本分布均衡，无需处理")
    else:
        stats = get_sample_stats(st.session_state.samples)
        st.warning(f"⚠️ 样本不均衡，最小/最大类别比例: {stats['balance_ratio']:.2f}")
        if st.button("⚖️ 应用SMOTE过采样", type='primary'):
            with st.spinner("正在执行SMOTE..."):
                try:
                    balanced_samples = apply_smote(st.session_state.samples)
                    st.session_state.samples = balanced_samples
                    new_stats = get_sample_stats(balanced_samples)
                    st.success(f"✅ SMOTE完成！\n\n原样本数: {stats['n_samples']}\n新样本数: {new_stats['n_samples']}\n新均衡比例: {new_stats['balance_ratio']:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ SMOTE失败: {str(e)}")

if st.session_state.samples.n_samples >= 2:
    st.markdown("---")
    st.subheader("📁 训练集/测试集划分")
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05)
    stratify = st.checkbox("分层采样（保持类别比例）", True)

    if st.button("✂️ 划分数据集", type='primary'):
        try:
            train_samples, test_samples = split_samples(st.session_state.samples, test_size=test_size, stratify=stratify)
            st.session_state.train_samples = train_samples
            st.session_state.test_samples = test_samples
            st.success(f"✅ 数据集划分完成！\n\n训练集: {train_samples.n_samples} 样本\n测试集: {test_samples.n_samples} 样本")

            col1, col2 = st.columns(2)
            with col1:
                train_stats = get_sample_stats(train_samples)
                st.markdown("#### 训练集分布")
                fig1 = px.bar(x=[train_stats['class_names'].get(c, f'Class {c}') for c in train_stats['class_counts'].keys()], y=list(train_stats['class_counts'].values()), title='训练集类别分布')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                test_stats = get_sample_stats(test_samples)
                st.markdown("#### 测试集分布")
                fig2 = px.bar(x=[test_stats['class_names'].get(c, f'Class {c}') for c in test_stats['class_counts'].keys()], y=list(test_stats['class_counts'].values()), title='测试集类别分布')
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"❌ 数据集划分失败: {str(e)}")
