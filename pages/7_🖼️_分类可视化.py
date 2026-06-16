import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import json
from PIL import Image

from state import init_session_state, render_sidebar_info
from src.visualization import (
    get_true_color, get_false_color, compute_ndvi, colormap_ndvi,
    get_rgb_composite, classification_to_rgb,
    overlay_classification, extract_single_class,
    generate_classification_legend
)


st.set_page_config(page_title="分类可视化", page_icon="🖼️", layout="wide")
init_session_state()
render_sidebar_info()

st.header("🖼️ 分类结果可视化")

if st.session_state.classification_result is None:
    st.warning("⚠️ 请先完成模型训练和分类")
    st.stop()

classification = st.session_state.classification_result
class_names = st.session_state.samples.class_names
legend = st.session_state.get('classification_legend', {})

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎨 分类图显示选项")
    display_mode = st.selectbox("显示模式", ["分类图", "叠加显示", "单类提取"])
    background_type = "真彩色"
    alpha = 0.5
    target_class = None

    if display_mode == "叠加显示":
        background_type = st.selectbox("背景影像", ["真彩色", "标准假彩色", "NDVI"])
        alpha = st.slider("分类图透明度", 0.1, 1.0, 0.5, 0.05)
    elif display_mode == "单类提取":
        classes = np.unique(classification)
        target_class = st.selectbox("选择要提取的类别", classes, format_func=lambda x: class_names.get(x, f'Class {x}'))

with col2:
    st.subheader("🌈 波段组合工具")
    band_combo = st.selectbox("预设组合", ["真彩色 (R-G-B)", "标准假彩色 (NIR-R-G)", "NDVI指数", "自定义"])
    r_band, g_band, b_band = 0, 0, 0
    if band_combo == "自定义":
        max_band = st.session_state.data.shape[2] - 1
        r_band = st.slider("R通道波段", 0, max_band, min(50, max_band))
        g_band = st.slider("G通道波段", 0, max_band, min(30, max_band))
        b_band = st.slider("B通道波段", 0, max_band, min(10, max_band))

st.markdown("---")

try:
    if band_combo == "真彩色 (R-G-B)":
        background = get_true_color(st.session_state.data, st.session_state.wavelengths)
    elif band_combo == "标准假彩色 (NIR-R-G)":
        background = get_false_color(st.session_state.data, st.session_state.wavelengths)
    elif band_combo == "NDVI指数":
        ndvi = compute_ndvi(st.session_state.data, st.session_state.wavelengths)
        background = colormap_ndvi(ndvi)
    else:
        background = get_rgb_composite(st.session_state.data, r_band, g_band, b_band)

    class_rgb = st.session_state.classification_rgb

    if display_mode == "分类图":
        display_img = class_rgb
        title = "地物分类图"
    elif display_mode == "叠加显示":
        if background_type == "真彩色":
            bg = get_true_color(st.session_state.data, st.session_state.wavelengths)
        elif background_type == "标准假彩色":
            bg = get_false_color(st.session_state.data, st.session_state.wavelengths)
        else:
            ndvi = compute_ndvi(st.session_state.data, st.session_state.wavelengths)
            bg = colormap_ndvi(ndvi)
        display_img = overlay_classification(bg, class_rgb, alpha=alpha)
        title = f"分类结果叠加 (透明度: {alpha})"
    else:
        bg = get_true_color(st.session_state.data, st.session_state.wavelengths)
        display_img = extract_single_class(classification, target_class, bg)
        title = f"{class_names.get(target_class, f'Class {target_class}')} 分布"

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.imshow(background)
        ax1.set_title('原始影像')
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.imshow(display_img)
        ax2.set_title(title)
        ax2.axis('off')
        st.pyplot(fig2)

    if legend:
        st.markdown("### 🏷️ 图例")
        st.markdown(generate_classification_legend(legend), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 分类结果统计")
    unique, counts = np.unique(classification, return_counts=True)
    total_pixels = classification.size
    stats_df = pd.DataFrame({
        '类别': [class_names.get(u, f'Class {u}') for u in unique],
        '像素数': counts,
        '占比': [f"{c/total_pixels*100:.2f}%" for c in counts]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(stats_df, use_container_width=True)
    with col2:
        fig = px.pie(values=counts, names=[class_names.get(u, f'Class {u}') for u in unique], title='地物类别分布')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💾 下载结果")
    col1, col2, col3 = st.columns(3)

    with col1:
        class_png = (class_rgb * 255).astype(np.uint8)
        img = Image.fromarray(class_png)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        st.download_button("📥 下载分类图 (PNG)", buf.getvalue(), "classification_map.png", "image/png")

    with col2:
        buf = io.BytesIO()
        np.save(buf, classification)
        st.download_button("📥 下载分类数据 (NPY)", buf.getvalue(), "classification_result.npy", "application/octet-stream")

    with col3:
        if st.session_state.metrics is not None:
            metrics_json = json.dumps({
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in st.session_state.metrics.items()
            }, ensure_ascii=False, indent=2)
            st.download_button("📥 下载评估指标 (JSON)", metrics_json, "metrics.json", "application/json")

except Exception as e:
    st.error(f"❌ 可视化失败: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
