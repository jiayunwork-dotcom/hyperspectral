import streamlit as st
import os
import sys
import tempfile
import numpy as np
from typing import Dict, List, Tuple

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
        'data_quality_report': None,
        'low_snr_bands': None,
        'hyperparam_results': None,
        'hyperparam_best_idx': None,
        'classification_results_history': {},
        'error_spatial_map': None,
        'class_confusion_pairs': None,
        'change_data_a': None,
        'change_header_a': None,
        'change_data_b': None,
        'change_header_b': None,
        'change_aligned_a': None,
        'change_aligned_b': None,
        'change_align_info': None,
        'change_method': None,
        'change_mask': None,
        'change_intensity': None,
        'change_stats': None,
        'change_binary_vis': None,
        'change_heat_vis': None,
        'change_class_a': None,
        'change_class_b': None,
        'transition_matrix': None,
        'transition_classes': None,
        'transition_stats': None,
        'sankey_data': None,
        'change_selected_pixel': None,
        'change_selected_region': None,
        'change_wavelengths_a': None,
        'change_wavelengths_b': None,
        'change_geojson': None,
        'multi_algo_comparison': None,
        'temporal_images': [],
        'temporal_headers': [],
        'temporal_wavelengths': [],
        'temporal_aligned': False,
        'temporal_analysis': None,
        'chord_data': None,
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


def compute_data_quality_report(data: np.ndarray, chunk_size: int = 1000) -> Dict:
    H, W, B = data.shape
    means = np.zeros(B, dtype=np.float64)
    stds = np.zeros(B, dtype=np.float64)
    n_pixels = 0

    from src.utils import chunk_generator
    for chunk, start, end in chunk_generator(data, chunk_size=chunk_size, axis=0):
        chunk = chunk.astype(np.float64)
        n_chunk = chunk.shape[0] * chunk.shape[1]
        means += np.sum(chunk.reshape(-1, B), axis=0)
        stds += np.sum(np.square(chunk.reshape(-1, B)), axis=0)
        n_pixels += n_chunk

    means /= n_pixels
    stds = np.sqrt(np.maximum(stds / n_pixels - np.square(means), 0))
    snr = means / (stds + 1e-10)

    return {
        'band_indices': np.arange(B),
        'means': means,
        'stds': stds,
        'snr': snr,
        'n_pixels': n_pixels,
        'n_bands': B
    }


def get_low_snr_bands(report: Dict, threshold: float) -> List[int]:
    return np.where(report['snr'] < threshold)[0].tolist()


def remove_bands_from_data(data: np.ndarray, band_indices: List[int]) -> np.ndarray:
    H, W, B = data.shape
    keep_indices = [i for i in range(B) if i not in band_indices]

    if isinstance(data, np.memmap):
        from src.utils import chunk_generator, save_temp_array, load_temp_array
        result = np.empty((H, W, len(keep_indices)), dtype=data.dtype)
        for chunk, start, end in chunk_generator(data, chunk_size=1000, axis=0):
            result[start:end] = chunk[:, :, keep_indices]
        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
        return result
    else:
        return data[:, :, keep_indices]


def save_classification_result(name: str, result: np.ndarray, rgb: np.ndarray, legend: Dict, metrics: Dict = None):
    st.session_state.classification_results_history[name] = {
        'result': result,
        'rgb': rgb,
        'legend': legend,
        'metrics': metrics,
        'classifier': st.session_state.classifier,
        'train_info': st.session_state.train_info
    }


def compute_class_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: Dict[int, str]) -> List[Dict]:
    from sklearn.metrics import confusion_matrix
    classes = np.unique(np.concatenate([y_true, y_pred]))
    labels = sorted(classes.tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                total_i = cm[i, :].sum()
                confusion_ratio = cm[i, j] / total_i if total_i > 0 else 0
                pairs.append({
                    'true_class': labels[i],
                    'true_name': class_names.get(labels[i], f'Class {labels[i]}'),
                    'pred_class': labels[j],
                    'pred_name': class_names.get(labels[j], f'Class {labels[j]}'),
                    'count': int(cm[i, j]),
                    'confusion_ratio': float(confusion_ratio)
                })

    pairs.sort(key=lambda x: x['confusion_ratio'], reverse=True)
    return pairs
