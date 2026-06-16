import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from state import init_session_state, create_progress_callback, render_sidebar_info
from src.preprocessing import preprocessing_pipeline
from src.visualization import plot_mnf_variance, plot_pca_variance
from src.utils import normalize_image


st.set_page_config(page_title="数据预处理", page_icon="⚙️", layout="wide")
init_session_state()
render_sidebar_info()

st.header("⚙️ 数据预处理")

if st.session_state.data is None:
    st.warning("⚠️ 请先导入数据")
    st.stop()

st.markdown("选择预处理步骤（按流水线顺序执行）：")

preprocess_steps = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 噪声波段剔除")
    enable_remove_bands = st.checkbox("启用噪声波段剔除", value=False)
    if enable_remove_bands:
        remove_mode = st.radio("剔除方式", ["自动检测（SNR阈值）", "手动指定波段"])
        if remove_mode == "自动检测（SNR阈值）":
            snr_threshold = st.slider("SNR阈值", 0.1, 2.0, 0.5, 0.1)
            preprocess_steps.append({'name': '噪声波段剔除', 'type': 'remove_bands', 'auto_detect': True, 'snr_threshold': snr_threshold})
        else:
            max_band = st.session_state.data.shape[2] - 1
            band_start = st.number_input("起始波段", 0, max_band, 0)
            band_end = st.number_input("结束波段", 0, max_band, max_band)
            band_indices = list(range(int(band_start), int(band_end) + 1))
            preprocess_steps.append({'name': '噪声波段剔除', 'type': 'remove_bands', 'auto_detect': False, 'band_indices': band_indices})

    st.subheader("2. 光谱平滑")
    enable_savgol = st.checkbox("启用Savitzky-Golay平滑", value=False)
    if enable_savgol:
        window_length = st.slider("窗口大小", 3, 15, 7, 2)
        polyorder = st.slider("多项式阶数", 1, 5, 3, 1)
        preprocess_steps.append({'name': '光谱平滑', 'type': 'savgol', 'window_length': window_length, 'polyorder': polyorder})

with col2:
    st.subheader("3. 降维变换")
    reduce_method = st.selectbox("选择降维方法", ["不使用", "MNF最小噪声分离", "PCA主成分分析"])
    if reduce_method == "MNF最小噪声分离":
        n_components = st.slider("保留MNF分量数", 2, 50, 10)
        preprocess_steps.append({'name': 'MNF变换', 'type': 'mnf', 'n_components': n_components, 'sample_size': 100000})
    elif reduce_method == "PCA主成分分析":
        variance_threshold = st.slider("累积方差阈值", 0.8, 0.99, 0.95, 0.01)
        preprocess_steps.append({'name': 'PCA变换', 'type': 'pca', 'variance_threshold': variance_threshold, 'sample_size': 100000})

if st.button("▶️ 执行预处理", type='primary'):
    if not preprocess_steps:
        st.warning("⚠️ 请至少选择一个预处理步骤")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    callback = create_progress_callback(progress_bar, status_text)

    try:
        with st.spinner("正在执行预处理..."):
            preprocessed_data, results = preprocessing_pipeline(
                st.session_state.data, preprocess_steps, progress_callback=callback
            )
            st.session_state.preprocessed_data = preprocessed_data
            st.session_state.preprocess_results = results
            progress_bar.progress(1.0)
            status_text.text("✅ 预处理完成！")
            st.success(f"✅ 预处理完成！\n\n原始尺寸: {st.session_state.data.shape}\n处理后尺寸: {preprocessed_data.shape}")

            if 'removed_bands' in results and results['removed_bands']:
                st.info(f"🎯 已剔除 {len(results['removed_bands'])} 个噪声波段")

            if 'mnf' in results:
                mnf_info = results['mnf']
                st.info(f"📊 MNF变换: 保留 {mnf_info['n_components']} 个分量\n累积方差贡献率: {mnf_info['cumulative_variance'][-1]:.4f}")
                var_data = plot_mnf_variance(mnf_info['explained_variance_ratio'])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=var_data['components'], y=var_data['individual'], name='单个方差'))
                fig.add_trace(go.Scatter(x=var_data['components'], y=var_data['cumulative'], mode='lines+markers', name='累积方差', yaxis='y2'))
                fig.update_layout(title='MNF方差解释率', xaxis_title='分量', yaxis_title='单个方差解释率', yaxis2=dict(title='累积方差解释率', overlaying='y', side='right'), height=400)
                st.plotly_chart(fig, use_container_width=True)

            if 'pca' in results:
                pca_info = results['pca']
                st.info(f"📊 PCA变换: 保留 {pca_info['n_components']} 个分量\n累积方差贡献率: {pca_info['cumulative_variance'][-1]:.4f}")
                var_data = plot_pca_variance(pca_info['explained_variance_ratio'])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=var_data['components'], y=var_data['individual'], name='单个方差'))
                fig.add_trace(go.Scatter(x=var_data['components'], y=var_data['cumulative'], mode='lines+markers', name='累积方差', yaxis='y2'))
                fig.update_layout(title='PCA方差解释率', xaxis_title='分量', yaxis_title='单个方差解释率', yaxis2=dict(title='累积方差解释率', overlaying='y', side='right'), height=400)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 预处理失败: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

if st.session_state.preprocessed_data is not None and st.session_state.preprocessed_data is not st.session_state.data:
    st.markdown("---")
    st.subheader("📊 预处理结果预览")
    preview_band = st.slider("选择分量", 0, st.session_state.preprocessed_data.shape[2] - 1, 0)

    col1, col2 = st.columns(2)
    with col1:
        img_orig = normalize_image(st.session_state.data[:, :, 0])
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.imshow(img_orig, cmap='gray')
        ax1.set_title('原始数据 (第1波段)')
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        img_proc = normalize_image(st.session_state.preprocessed_data[:, :, preview_band])
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(img_proc, cmap='gray')
        ax2.set_title(f'预处理后 (第{preview_band}分量)')
        ax2.axis('off')
        st.pyplot(fig2)
