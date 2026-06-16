import streamlit as st
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sample_management import create_empty_samples
from src.data_io import get_image_info


def init_session_state():
    defaults = {
        'data': None,
        'header': None,
        'preprocessed_data': None,
        'features': None,
        'samples': create_empty_samples(),
        'classifier': None,
        'classification_result': None,
        'classification_rgb': None,
        'classification_legend': {},
        'metrics': None,
        'wavelengths': None,
        'preprocess_results': None,
        'feature_info': None,
        'train_samples': None,
        'test_samples': None,
        'y_true': None,
        'y_pred': None,
        'train_info': None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def save_uploaded_file(uploaded_file, suffix: str = '') -> str:
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def show_progress(progress_bar, status_text, progress: float, message: str):
    if progress_bar is not None:
        progress_bar.progress(progress)
    if status_text is not None:
        status_text.text(message)


def create_progress_callback(progress_bar, status_text):
    return lambda p, m: show_progress(progress_bar, status_text, p, m)


def render_sidebar_info():
    if st.session_state.header is not None:
        info = get_image_info(st.session_state.data, st.session_state.header)
        st.sidebar.subheader("影像基本信息")
        st.sidebar.info(
            f"📐 尺寸: {info['dimensions']}\n\n"
            f"🎯 波段数: {info['num_bands']}\n\n"
            f"💾 数据类型: {info['data_type']}\n\n"
            f"📦 内存占用: {info['memory_mb']:.1f} MB\n\n"
            f"🔀 存储格式: {info['interleave']}\n\n"
            + (f"🌈 波长范围: {info['wavelength_range'][0]:.0f}-{info['wavelength_range'][1]:.0f} nm" if info['wavelength_range'] else "")
        )
        if info['map_info']:
            st.sidebar.info(f"🗺️ 坐标信息: {info['map_info'][:50]}...")
