import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from state import (
    init_session_state, save_uploaded_file, render_sidebar_info,
    compute_data_quality_report, get_low_snr_bands, remove_bands_from_data
)
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
                    
                    with st.spinner("正在计算数据质量报告..."):
                        st.session_state.data_quality_report = compute_data_quality_report(data)
                    
                    st.success(f"✅ 数据加载成功！\n\n尺寸: {header.lines} × {header.samples} 像素\n波段数: {header.bands}\n数据类型: {data.dtype}")
                except Exception as e:
                    st.error(f"❌ 数据加载失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

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

    if st.session_state.data_quality_report is not None:
        st.markdown("---")
        st.subheader("📋 数据质量报告")

        report = st.session_state.data_quality_report
        col1, col2 = st.columns([1, 1])

        with col1:
            snr_threshold = st.slider(
                "信噪比(SNR)阈值",
                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                help="低于此阈值的波段将被标记为低质量波段"
            )
            low_snr_bands = get_low_snr_bands(report, snr_threshold)
            st.session_state.low_snr_bands = low_snr_bands

            st.metric("总波段数", report['n_bands'])
            st.metric("低SNR波段数", len(low_snr_bands),
                      delta=f"-{len(low_snr_bands)/report['n_bands']*100:.1f}%" if len(low_snr_bands) > 0 else "正常")

            if len(low_snr_bands) > 0:
                st.warning(f"⚠️ 检测到 {len(low_snr_bands)} 个低信噪比波段")
                if st.button("🗑️ 一键剔除低SNR波段", type='primary'):
                    with st.spinner("正在剔除低质量波段..."):
                        try:
                            new_data = remove_bands_from_data(st.session_state.data, low_snr_bands)
                            new_preprocessed = remove_bands_from_data(st.session_state.preprocessed_data, low_snr_bands)

                            new_wavelengths = None
                            if st.session_state.wavelengths is not None:
                                keep_mask = np.ones(len(st.session_state.wavelengths), dtype=bool)
                                keep_mask[low_snr_bands] = False
                                new_wavelengths = st.session_state.wavelengths[keep_mask]

                            st.session_state.data = new_data
                            st.session_state.preprocessed_data = new_preprocessed
                            st.session_state.wavelengths = new_wavelengths
                            st.session_state.data_quality_report = compute_data_quality_report(new_data)
                            st.session_state.low_snr_bands = None

                            st.success(f"✅ 已成功剔除 {len(low_snr_bands)} 个低质量波段！\n\n剩余波段数: {new_data.shape[2]}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 剔除失败: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        with col2:
            wavelengths_plot = st.session_state.wavelengths if st.session_state.wavelengths is not None else report['band_indices']

            fig = go.Figure()
            colors = ['red' if s < snr_threshold else 'blue' for s in report['snr']]
            fig.add_trace(go.Scatter(
                x=wavelengths_plot,
                y=report['snr'],
                mode='lines+markers',
                marker=dict(color=colors, size=4),
                line=dict(color='rgba(100,100,100,0.5)'),
                name='SNR'
            ))
            fig.add_hline(
                y=snr_threshold, line_dash="dash", line_color="red",
                annotation_text=f"阈值: {snr_threshold}", annotation_position="top right"
            )
            fig.update_layout(
                title='各波段信噪比(SNR)分布',
                xaxis_title='波长 (nm)' if st.session_state.wavelengths is not None else '波段索引',
                yaxis_title='SNR',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        tab1, tab2 = st.tabs(["📈 均值与标准差", "📊 详细统计表格"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wavelengths_plot, y=report['means'],
                mode='lines', name='均值', yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=wavelengths_plot, y=report['stds'],
                mode='lines', name='标准差', yaxis='y2'
            ))
            fig.update_layout(
                title='各波段均值与标准差分布',
                xaxis_title='波长 (nm)' if st.session_state.wavelengths is not None else '波段索引',
                yaxis_title='均值',
                yaxis2=dict(title='标准差', overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            quality_df = pd.DataFrame({
                '波段索引': report['band_indices'],
                '波长 (nm)': st.session_state.wavelengths if st.session_state.wavelengths is not None else ['N/A'] * len(report['band_indices']),
                '均值': [f"{m:.6f}" for m in report['means']],
                '标准差': [f"{s:.6f}" for s in report['stds']],
                'SNR': [f"{s:.4f}" for s in report['snr']],
                '质量状态': ['⚠️ 低SNR' if s < snr_threshold else '✅ 正常' for s in report['snr']]
            })

            def highlight_low_snr(row):
                return ['background-color: #ffe6e6' if '低SNR' in str(row['质量状态']) else '' for _ in row]

            st.dataframe(
                quality_df.style.apply(highlight_low_snr, axis=1),
                use_container_width=True,
                height=400
            )
