import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from state import init_session_state, save_uploaded_file, render_sidebar_info
from src.data_io import load_envi_data, load_envi_labels, get_image_info
from src.sample_management import extract_labeled_samples
from src.visualization import get_true_color, get_false_color, compute_ndvi, colormap_ndvi
from src.utils import normalize_image


init_session_state()
render_sidebar_info()

st.header("📥 数据导入")
st.markdown("支持ENVI格式的高光谱数据，需要同时上传 `.hdr` 头文件和对应的二进制数据文件。")

col1, col2 = st.columns(2)

with col1:
    st.subheader("上传数据文件")
    hdr_file = st.file_uploader("选择头文件 (.hdr)", type=['hdr'])
    dat_file = st.file_uploader("选择数据文件")

    if hdr_file and dat_file:
        if st.button("🚀 加载数据", type='primary'):
            with st.spinner("正在加载高光谱数据..."):
                try:
                    hdr_path = save_uploaded_file(hdr_file, '.hdr')
                    dat_path = save_uploaded_file(dat_file, '')
                    data, header = load_envi_data(hdr_path, dat_path, mmap=True)
                    st.session_state.data = data
                    st.session_state.header = header
                    st.session_state.preprocessed_data = data
                    st.session_state.wavelengths = np.array(header.wavelengths) if header.wavelengths else None
                    st.success(f"✅ 数据加载成功！\n\n尺寸: {header.lines} × {header.samples} 像素\n波段数: {header.bands}\n数据类型: {data.dtype}")
                except Exception as e:
                    st.error(f"❌ 数据加载失败: {str(e)}")

with col2:
    st.subheader("上传标签文件 (可选)")
    label_file = st.file_uploader("选择地面标签文件", type=['npy', 'tif', 'tiff', 'png', 'hdr'])
    label_hdr_file = st.file_uploader("选择标签头文件 (.hdr)", type=['hdr'])

    if label_file and st.session_state.data is not None:
        if st.button("📝 加载标签"):
            with st.spinner("正在加载标签数据..."):
                try:
                    label_path = save_uploaded_file(label_file)
                    label_hdr_path = save_uploaded_file(label_hdr_file, '.hdr') if label_hdr_file else None
                    labels = load_envi_labels(label_path, label_hdr_path)
                    if labels.shape[:2] != st.session_state.data.shape[:2]:
                        st.warning(f"⚠️ 标签尺寸 {labels.shape} 与影像尺寸 {st.session_state.data.shape[:2]} 不匹配")
                    else:
                        st.session_state.samples = extract_labeled_samples(st.session_state.preprocessed_data, labels)
                        st.success(f"✅ 标签加载成功！\n\n样本数: {st.session_state.samples.n_samples}\n类别数: {len(st.session_state.samples.classes)}")
                except Exception as e:
                    st.error(f"❌ 标签加载失败: {str(e)}")

if st.session_state.data is not None:
    st.markdown("---")
    st.subheader("🔍 数据预览")

    preview_type = st.selectbox("选择预览方式", ["真彩色合成", "标准假彩色", "NDVI指数", "单波段显示"])
    wavelengths = st.session_state.wavelengths

    try:
        if preview_type == "真彩色合成":
            img = get_true_color(st.session_state.data, wavelengths)
        elif preview_type == "标准假彩色":
            img = get_false_color(st.session_state.data, wavelengths)
        elif preview_type == "NDVI指数":
            ndvi = compute_ndvi(st.session_state.data, wavelengths)
            img = colormap_ndvi(ndvi)
        else:
            band_idx = st.slider("选择波段", 0, st.session_state.data.shape[2] - 1, 0)
            img = normalize_image(st.session_state.data[:, :, band_idx])
            img = np.stack([img] * 3, axis=-1)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(f"{preview_type}预览")
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"预览生成失败: {str(e)}")

    if st.session_state.wavelengths is not None and len(st.session_state.wavelengths) > 0:
        st.markdown("---")
        st.subheader("📊 光谱曲线预览")
        n_samples = st.session_state.data.shape[0] * st.session_state.data.shape[1]
        random_indices = np.random.choice(n_samples, 5, replace=False)
        data_flat = st.session_state.data.reshape(-1, st.session_state.data.shape[2])

        fig = go.Figure()
        for i, idx in enumerate(random_indices):
            spectrum = data_flat[idx]
            fig.add_trace(go.Scatter(x=st.session_state.wavelengths, y=spectrum, mode='lines', name=f'随机像素 {i+1}'))
        fig.update_layout(title='随机像素光谱曲线', xaxis_title='波长 (nm)', yaxis_title='反射值', height=400)
        st.plotly_chart(fig, use_container_width=True)
