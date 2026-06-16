import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from state import init_session_state, create_progress_callback, render_sidebar_info
from src.feature_extraction import extract_features
from src.utils import normalize_image


init_session_state()
render_sidebar_info()

st.header("🔬 特征提取")

if st.session_state.preprocessed_data is None:
    st.warning("⚠️ 请先完成数据导入和预处理")
    st.stop()

feature_type = st.selectbox("选择特征类型", ["光谱特征", "空间特征", "光谱+空间融合特征"])

col1, col2 = st.columns(2)

with col1:
    spectral_features = []
    if feature_type in ["光谱特征", "光谱+空间融合特征"]:
        st.subheader("光谱特征选项")
        spectral_features = st.multiselect(
            "选择光谱特征",
            ['continuum_removal', 'first_derivative', 'second_derivative', 'absorption_peaks'],
            default=['continuum_removal', 'first_derivative'],
            format_func=lambda x: {'continuum_removal': '连续统去除', 'first_derivative': '一阶导数', 'second_derivative': '二阶导数', 'absorption_peaks': '吸收峰检测'}[x]
        )

with col2:
    spatial_features = []
    mp_scales = None
    gabor_freqs = None
    if feature_type in ["空间特征", "光谱+空间融合特征"]:
        st.subheader("空间特征选项")
        spatial_features = st.multiselect(
            "选择空间特征",
            ['morphological_profile', 'gabor'],
            default=['morphological_profile', 'gabor'],
            format_func=lambda x: {'morphological_profile': '形态学剖面', 'gabor': 'Gabor纹理特征'}[x]
        )
        if 'morphological_profile' in spatial_features:
            mp_scales = st.multiselect("形态学尺度", [3, 5, 7, 9, 11, 13], default=[3, 5, 7])
        if 'gabor' in spatial_features:
            gabor_freqs = st.multiselect("Gabor频率", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.1, 0.2, 0.3])

if st.button("🚀 提取特征", type='primary'):
    if feature_type == "光谱特征" and not spectral_features:
        st.warning("⚠️ 请至少选择一个光谱特征")
        st.stop()
    if feature_type == "空间特征" and not spatial_features:
        st.warning("⚠️ 请至少选择一个空间特征")
        st.stop()

    feature_type_map = {"光谱特征": 'spectral', "空间特征": 'spatial', "光谱+空间融合特征": 'fused'}
    progress_bar = st.progress(0)
    status_text = st.empty()
    callback = create_progress_callback(progress_bar, status_text)

    try:
        with st.spinner("正在提取特征..."):
            features, feature_info = extract_features(
                st.session_state.preprocessed_data,
                feature_type=feature_type_map[feature_type],
                wavelengths=st.session_state.wavelengths,
                spectral_features=spectral_features if spectral_features else None,
                spatial_features=spatial_features if spatial_features else None,
                mp_scales=mp_scales,
                gabor_frequencies=gabor_freqs,
                progress_callback=callback
            )
            st.session_state.features = features
            st.session_state.feature_info = feature_info
            progress_bar.progress(1.0)
            status_text.text("✅ 特征提取完成！")
            st.success(f"✅ 特征提取完成！\n\n特征维度: {features.shape}\n特征数: {feature_info['n_features']}")
            if 'feature_names' in feature_info:
                st.info(f"📋 特征列表: {', '.join(feature_info['feature_names'][:10])}...")
    except Exception as e:
        st.error(f"❌ 特征提取失败: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

if st.session_state.features is not None:
    st.markdown("---")
    st.subheader("📊 特征可视化")
    feature_band = st.slider("选择特征分量", 0, st.session_state.features.shape[2] - 1, 0)
    img = normalize_image(st.session_state.features[:, :, feature_band])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(img, cmap='viridis')
    ax.set_title(f'特征分量 {feature_band}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
