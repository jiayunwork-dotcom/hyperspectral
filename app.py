import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_io import load_envi_data, load_envi_labels, get_image_info, ENVIHeader
from src.preprocessing import (
    preprocessing_pipeline, remove_noisy_bands, mnf_transform,
    pca_transform, savgol_smoothing, compute_band_snr
)
from src.feature_extraction import extract_features
from src.classification import (
    create_classifier, classify_image, train_test_split,
    SVMClassifier, RandomForestClassifierHSI, OneDCNNClassifier,
    ThreeDCNNClassifier, SemiSupervisedClassifier
)
from src.sample_management import (
    TrainingSamples, ROIRegion, extract_labeled_samples,
    add_roi_to_samples, apply_smote, split_samples,
    get_sample_stats, create_empty_samples
)
from src.evaluation import (
    compute_metrics, compute_confusion_matrix,
    evaluate_classifier, format_metrics_for_display
)
from src.visualization import (
    get_rgb_composite, get_true_color, get_false_color,
    compute_ndvi, colormap_ndvi, classification_to_rgb,
    overlay_classification, extract_single_class,
    plot_spectrum, plot_mnf_variance, plot_pca_variance,
    plot_confusion_matrix, plot_class_distribution,
    generate_classification_legend
)
from src.batch_processing import create_batch_jobs, batch_process, BatchJob


st.set_page_config(
    page_title="高光谱遥感影像分类系统",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'data' not in st.session_state:
    st.session_state.data = None
if 'header' not in st.session_state:
    st.session_state.header = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'samples' not in st.session_state:
    st.session_state.samples = create_empty_samples()
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'wavelengths' not in st.session_state:
    st.session_state.wavelengths = None
if 'preprocess_results' not in st.session_state:
    st.session_state.preprocess_results = None
if 'feature_info' not in st.session_state:
    st.session_state.feature_info = None
if 'train_samples' not in st.session_state:
    st.session_state.train_samples = None
if 'test_samples' not in st.session_state:
    st.session_state.test_samples = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = '数据导入'
if 'batch_files' not in st.session_state:
    st.session_state.batch_files = []


PAGES = ['数据导入', '数据预处理', '特征提取', '样本管理', '模型训练', '精度评估', '分类可视化', '批量处理']


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


def sidebar_navigation():
    st.sidebar.title("🛰️ 高光谱分类系统")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("功能模块", PAGES, index=PAGES.index(st.session_state.current_page))
    st.session_state.current_page = page

    st.sidebar.markdown("---")

    if st.session_state.header is not None:
        st.sidebar.subheader("影像基本信息")
        info = get_image_info(st.session_state.data, st.session_state.header)
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

    return page


def page_data_import():
    st.header("📥 数据导入")
    st.markdown("支持ENVI格式的高光谱数据，需要同时上传 `.hdr` 头文件和对应的二进制数据文件。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("上传数据文件")
        hdr_file = st.file_uploader("选择头文件 (.hdr)", type=['hdr'], key='hdr_upload')
        dat_file = st.file_uploader("选择数据文件", type=[''], key='dat_upload')

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

                        st.success(f"✅ 数据加载成功！\n\n"
                                   f"尺寸: {header.lines} × {header.samples} 像素\n"
                                   f"波段数: {header.bands}\n"
                                   f"数据类型: {data.dtype}")

                    except Exception as e:
                        st.error(f"❌ 数据加载失败: {str(e)}")

    with col2:
        st.subheader("上传标签文件 (可选)")
        label_file = st.file_uploader("选择地面标签文件", type=['npy', 'tif', 'tiff', 'png', 'hdr'], key='label_upload')
        label_hdr_file = st.file_uploader("选择标签头文件 (.hdr)", type=['hdr'], key='label_hdr_upload')

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
                            st.session_state.samples = extract_labeled_samples(
                                st.session_state.preprocessed_data, labels
                            )
                            st.success(f"✅ 标签加载成功！\n\n"
                                       f"样本数: {st.session_state.samples.n_samples}\n"
                                       f"类别数: {len(st.session_state.samples.classes)}")

                    except Exception as e:
                        st.error(f"❌ 标签加载失败: {str(e)}")

    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("🔍 数据预览")

        preview_type = st.selectbox(
            "选择预览方式",
            ["真彩色合成", "标准假彩色", "NDVI指数", "单波段显示"],
            key='preview_type'
        )

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
                img = st.session_state.data[:, :, band_idx]
                from src.utils import normalize_image
                img = normalize_image(img)
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
                fig.add_trace(go.Scatter(
                    x=st.session_state.wavelengths,
                    y=spectrum,
                    mode='lines',
                    name=f'随机像素 {i+1}'
                ))

            fig.update_layout(
                title='随机像素光谱曲线',
                xaxis_title='波长 (nm)',
                yaxis_title='反射值',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def page_preprocessing():
    st.header("⚙️ 数据预处理")

    if st.session_state.data is None:
        st.warning("⚠️ 请先导入数据")
        return

    st.markdown("选择预处理步骤（按流水线顺序执行）：")

    preprocess_steps = []

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. 噪声波段剔除")
        enable_remove_bands = st.checkbox("启用噪声波段剔除", value=False, key='enable_remove_bands')

        if enable_remove_bands:
            remove_mode = st.radio("剔除方式", ["自动检测（SNR阈值）", "手动指定波段"], key='remove_mode')

            if remove_mode == "自动检测（SNR阈值）":
                snr_threshold = st.slider("SNR阈值", 0.1, 2.0, 0.5, 0.1, key='snr_threshold')
                preprocess_steps.append({
                    'name': '噪声波段剔除',
                    'type': 'remove_bands',
                    'auto_detect': True,
                    'snr_threshold': snr_threshold
                })
            else:
                max_band = st.session_state.data.shape[2] - 1
                band_start = st.number_input("起始波段", 0, max_band, 0, key='band_start')
                band_end = st.number_input("结束波段", 0, max_band, max_band, key='band_end')
                band_indices = list(range(int(band_start), int(band_end) + 1))
                preprocess_steps.append({
                    'name': '噪声波段剔除',
                    'type': 'remove_bands',
                    'auto_detect': False,
                    'band_indices': band_indices
                })

        st.subheader("2. 光谱平滑")
        enable_savgol = st.checkbox("启用Savitzky-Golay平滑", value=False, key='enable_savgol')

        if enable_savgol:
            window_length = st.slider("窗口大小", 3, 15, 7, 2, key='sg_window')
            polyorder = st.slider("多项式阶数", 1, 5, 3, 1, key='sg_order')
            preprocess_steps.append({
                'name': '光谱平滑',
                'type': 'savgol',
                'window_length': window_length,
                'polyorder': polyorder
            })

    with col2:
        st.subheader("3. 降维变换")
        reduce_method = st.selectbox(
            "选择降维方法",
            ["不使用", "MNF最小噪声分离", "PCA主成分分析"],
            key='reduce_method'
        )

        if reduce_method == "MNF最小噪声分离":
            n_components = st.slider("保留MNF分量数", 2, 50, 10, key='mnf_components')
            preprocess_steps.append({
                'name': 'MNF变换',
                'type': 'mnf',
                'n_components': n_components,
                'sample_size': 100000
            })

        elif reduce_method == "PCA主成分分析":
            variance_threshold = st.slider("累积方差阈值", 0.8, 0.99, 0.95, 0.01, key='pca_variance')
            preprocess_steps.append({
                'name': 'PCA变换',
                'type': 'pca',
                'variance_threshold': variance_threshold,
                'sample_size': 100000
            })

    if st.button("▶️ 执行预处理", type='primary'):
        if not preprocess_steps:
            st.warning("⚠️ 请至少选择一个预处理步骤")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        callback = create_progress_callback(progress_bar, status_text)

        try:
            with st.spinner("正在执行预处理..."):
                preprocessed_data, results = preprocessing_pipeline(
                    st.session_state.data,
                    preprocess_steps,
                    progress_callback=callback
                )

                st.session_state.preprocessed_data = preprocessed_data
                st.session_state.preprocess_results = results

                progress_bar.progress(1.0)
                status_text.text("✅ 预处理完成！")

                st.success(f"✅ 预处理完成！\n\n"
                           f"原始尺寸: {st.session_state.data.shape}\n"
                           f"处理后尺寸: {preprocessed_data.shape}")

                if 'removed_bands' in results and results['removed_bands']:
                    st.info(f"🎯 已剔除 {len(results['removed_bands'])} 个噪声波段")

                if 'mnf' in results:
                    mnf_info = results['mnf']
                    st.info(f"📊 MNF变换: 保留 {mnf_info['n_components']} 个分量\n"
                            f"累积方差贡献率: {mnf_info['cumulative_variance'][-1]:.4f}")

                    var_data = plot_mnf_variance(mnf_info['explained_variance_ratio'])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=var_data['components'],
                        y=var_data['individual'],
                        name='单个方差'
                    ))
                    fig.add_trace(go.Scatter(
                        x=var_data['components'],
                        y=var_data['cumulative'],
                        mode='lines+markers',
                        name='累积方差',
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        title='MNF方差解释率',
                        xaxis_title='分量',
                        yaxis_title='单个方差解释率',
                        yaxis2=dict(title='累积方差解释率', overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if 'pca' in results:
                    pca_info = results['pca']
                    st.info(f"📊 PCA变换: 保留 {pca_info['n_components']} 个分量\n"
                            f"累积方差贡献率: {pca_info['cumulative_variance'][-1]:.4f}")

                    var_data = plot_pca_variance(pca_info['explained_variance_ratio'])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=var_data['components'],
                        y=var_data['individual'],
                        name='单个方差'
                    ))
                    fig.add_trace(go.Scatter(
                        x=var_data['components'],
                        y=var_data['cumulative'],
                        mode='lines+markers',
                        name='累积方差',
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        title='PCA方差解释率',
                        xaxis_title='分量',
                        yaxis_title='单个方差解释率',
                        yaxis2=dict(title='累积方差解释率', overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 预处理失败: {str(e)}")
            progress_bar.progress(0)
            status_text.text("")

    if st.session_state.preprocessed_data is not None and st.session_state.preprocessed_data is not st.session_state.data:
        st.markdown("---")
        st.subheader("📊 预处理结果预览")

        preview_band = st.slider("选择分量", 0, st.session_state.preprocessed_data.shape[2] - 1, 0, key='preview_band')

        col1, col2 = st.columns(2)

        with col1:
            from src.utils import normalize_image
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


def page_feature_extraction():
    st.header("🔬 特征提取")

    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ 请先完成数据导入和预处理")
        return

    feature_type = st.selectbox(
        "选择特征类型",
        ["光谱特征", "空间特征", "光谱+空间融合特征"],
        key='feature_type'
    )

    col1, col2 = st.columns(2)

    with col1:
        if feature_type in ["光谱特征", "光谱+空间融合特征"]:
            st.subheader("光谱特征选项")
            spectral_features = st.multiselect(
                "选择光谱特征",
                ['continuum_removal', 'first_derivative', 'second_derivative', 'absorption_peaks'],
                default=['continuum_removal', 'first_derivative'],
                format_func=lambda x: {
                    'continuum_removal': '连续统去除',
                    'first_derivative': '一阶导数',
                    'second_derivative': '二阶导数',
                    'absorption_peaks': '吸收峰检测'
                }[x],
                key='spectral_features'
            )
        else:
            spectral_features = []

    with col2:
        if feature_type in ["空间特征", "光谱+空间融合特征"]:
            st.subheader("空间特征选项")
            spatial_features = st.multiselect(
                "选择空间特征",
                ['morphological_profile', 'gabor'],
                default=['morphological_profile', 'gabor'],
                format_func=lambda x: {
                    'morphological_profile': '形态学剖面',
                    'gabor': 'Gabor纹理特征'
                }[x],
                key='spatial_features'
            )

            if 'morphological_profile' in spatial_features:
                mp_scales = st.multiselect(
                    "形态学尺度",
                    [3, 5, 7, 9, 11, 13],
                    default=[3, 5, 7],
                    key='mp_scales'
                )
            else:
                mp_scales = None

            if 'gabor' in spatial_features:
                gabor_freqs = st.multiselect(
                    "Gabor频率",
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    default=[0.1, 0.2, 0.3],
                    key='gabor_freqs'
                )
            else:
                gabor_freqs = None
        else:
            spatial_features = []
            mp_scales = None
            gabor_freqs = None

    if st.button("🚀 提取特征", type='primary'):
        if feature_type == "光谱特征" and not spectral_features:
            st.warning("⚠️ 请至少选择一个光谱特征")
            return
        if feature_type == "空间特征" and not spatial_features:
            st.warning("⚠️ 请至少选择一个空间特征")
            return

        feature_type_map = {
            "光谱特征": 'spectral',
            "空间特征": 'spatial',
            "光谱+空间融合特征": 'fused'
        }

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

                st.success(f"✅ 特征提取完成！\n\n"
                           f"特征维度: {features.shape}\n"
                           f"特征数: {feature_info['n_features']}")

                if 'feature_names' in feature_info:
                    st.info(f"📋 特征列表: {', '.join(feature_info['feature_names'][:10])}...")

        except Exception as e:
            st.error(f"❌ 特征提取失败: {str(e)}")
            progress_bar.progress(0)
            status_text.text("")

    if st.session_state.features is not None:
        st.markdown("---")
        st.subheader("📊 特征可视化")

        feature_band = st.slider("选择特征分量", 0, st.session_state.features.shape[2] - 1, 0, key='feature_band')

        from src.utils import normalize_image
        img = normalize_image(st.session_state.features[:, :, feature_band])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(f'特征分量 {feature_band}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)


def page_sample_management():
    st.header("📝 训练样本管理")

    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ 请先完成数据导入和预处理")
        return

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
            dist_data = plot_class_distribution(
                stats['class_counts'],
                stats['class_names']
            )
            fig = px.bar(
                x=dist_data['labels'],
                y=dist_data['counts'],
                text=dist_data['counts'],
                title='样本类别分布'
            )
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

        from src.utils import normalize_image
        preview_img = get_true_color(st.session_state.data, st.session_state.wavelengths)

        st.image(preview_img, caption="影像预览（用于参考ROI位置）", use_column_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ROI设置")
            roi_label = st.number_input("类别标签", 1, 100, 1, key='roi_label')
            roi_name = st.text_input("类别名称", f"类别 {roi_label}", key='roi_name')
            shape_type = st.selectbox("ROI形状", ["rectangle", "polygon"], key='shape_type',
                                     format_func=lambda x: '矩形' if x == 'rectangle' else '多边形')

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
                coords_text = st.text_area(
                    "顶点坐标",
                    "10, 10\n10, 50\n30, 30",
                    height=100,
                    key='polygon_coords'
                )
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
                return

            roi = ROIRegion(
                label=int(roi_label),
                label_name=roi_name,
                coordinates=coordinates,
                shape_type=shape_type
            )

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
                '序号': i + 1,
                '标签': roi.label,
                '名称': roi.label_name,
                '形状': '矩形' if roi.shape_type == 'rectangle' else '多边形',
                '像素数': len(roi.pixels)
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
            st.warning(f"⚠️ 样本不均衡，最小/最大类别比例: {stats['balance_ratio']:.2f}")

            if st.button("⚖️ 应用SMOTE过采样", type='primary'):
                with st.spinner("正在执行SMOTE..."):
                    try:
                        balanced_samples = apply_smote(st.session_state.samples)
                        st.session_state.samples = balanced_samples

                        new_stats = get_sample_stats(balanced_samples)
                        st.success(f"✅ SMOTE完成！\n\n"
                                   f"原样本数: {stats['n_samples']}\n"
                                   f"新样本数: {new_stats['n_samples']}\n"
                                   f"新均衡比例: {new_stats['balance_ratio']:.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ SMOTE失败: {str(e)}")

    if st.session_state.samples.n_samples >= 2:
        st.markdown("---")
        st.subheader("📁 训练集/测试集划分")

        test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05, key='test_size')
        stratify = st.checkbox("分层采样（保持类别比例）", True, key='stratify')

        if st.button("✂️ 划分数据集", type='primary'):
            try:
                train_samples, test_samples = split_samples(
                    st.session_state.samples,
                    test_size=test_size,
                    stratify=stratify
                )

                st.session_state.train_samples = train_samples
                st.session_state.test_samples = test_samples

                st.success(f"✅ 数据集划分完成！\n\n"
                           f"训练集: {train_samples.n_samples} 样本\n"
                           f"测试集: {test_samples.n_samples} 样本")

                col1, col2 = st.columns(2)

                with col1:
                    train_stats = get_sample_stats(train_samples)
                    st.markdown("#### 训练集分布")
                    fig1 = px.bar(
                        x=[train_stats['class_names'].get(c, f'Class {c}') for c in train_stats['class_counts'].keys()],
                        y=list(train_stats['class_counts'].values()),
                        title='训练集类别分布'
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    test_stats = get_sample_stats(test_samples)
                    st.markdown("#### 测试集分布")
                    fig2 = px.bar(
                        x=[test_stats['class_names'].get(c, f'Class {c}') for c in test_stats['class_counts'].keys()],
                        y=list(test_stats['class_counts'].values()),
                        title='测试集类别分布'
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 数据集划分失败: {str(e)}")


def page_model_training():
    st.header("🤖 模型训练")

    if st.session_state.train_samples is None or st.session_state.test_samples is None:
        st.warning("⚠️ 请先完成样本管理和数据集划分")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("选择分类器")
        classifier_type = st.selectbox(
            "分类算法",
            ["SVM (RBF核)", "随机森林", "1D-CNN", "3D-CNN", "半监督学习（标签传播）"],
            key='classifier_type'
        )

    with col2:
        st.subheader("算法参数")

        if classifier_type == "SVM (RBF核)":
            st.markdown("##### SVM参数")
            enable_grid_search = st.checkbox("启用网格搜索优化", True, key='grid_search')
            C_values = st.multiselect(
                "C值候选",
                [0.01, 0.1, 1, 10, 100],
                default=[0.1, 1, 10],
                key='C_values'
            )
            gamma_values = st.multiselect(
                "gamma候选",
                ['scale', 0.001, 0.01, 0.1, 1],
                default=['scale', 0.01, 0.1],
                key='gamma_values'
            )
            classifier_params = {
                'C': C_values,
                'gamma': gamma_values,
                'grid_search': enable_grid_search,
                'cv': 3
            }
            classifier_key = 'svm'

        elif classifier_type == "随机森林":
            st.markdown("##### 随机森林参数")
            n_estimators = st.slider("决策树数量", 10, 500, 100, 10, key='n_estimators')
            max_depth = st.slider("最大深度", 3, 30, 10, key='max_depth')
            max_features = st.selectbox("特征采样比例", ['sqrt', 'log2', 0.3, 0.5, 0.7], key='max_features')
            if isinstance(max_features, str):
                max_features_param = max_features
            else:
                max_features_param = float(max_features)
            classifier_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features_param
            }
            classifier_key = 'random_forest'

        elif classifier_type == "1D-CNN":
            st.markdown("##### 1D-CNN参数")
            n_epochs = st.slider("训练轮数", 10, 200, 50, 10, key='n_epochs')
            batch_size = st.selectbox("批大小", [32, 64, 128, 256], 2, key='batch_size')
            learning_rate = st.selectbox("学习率", [0.0001, 0.001, 0.01, 0.1], 1, key='learning_rate')
            classifier_params = {
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            classifier_key = '1d_cnn'

        elif classifier_type == "3D-CNN":
            st.markdown("##### 3D-CNN参数")
            window_size = st.selectbox("空间窗口大小", [3, 5, 7, 9, 11], 2, key='window_size')
            n_epochs = st.slider("训练轮数", 10, 100, 30, 5, key='n_epochs_3d')
            batch_size = st.selectbox("批大小", [32, 64, 128, 256], 1, key='batch_size_3d')
            learning_rate = st.selectbox("学习率", [0.0001, 0.001, 0.01], 1, key='learning_rate_3d')
            classifier_params = {
                'window_size': window_size,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            classifier_key = '3d_cnn'

        else:
            st.markdown("##### 半监督学习参数")
            gamma = st.slider("RBF核gamma", 1, 50, 20, key='gamma_semi')
            max_iter = st.slider("最大迭代次数", 100, 5000, 1000, 100, key='max_iter')
            n_neighbors = st.slider("邻居数", 3, 15, 7, key='n_neighbors')

            use_unlabeled = st.checkbox("使用无标注样本", True, key='use_unlabeled')
            classifier_params = {
                'gamma': gamma,
                'max_iter': max_iter,
                'n_neighbors': n_neighbors
            }
            classifier_key = 'semi_supervised'

    if st.button("🚀 开始训练", type='primary'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        callback = create_progress_callback(progress_bar, status_text)

        try:
            with st.spinner("正在训练模型..."):
                classifier = create_classifier(classifier_key, **classifier_params)

                train_X = st.session_state.train_samples.features
                train_y = st.session_state.train_samples.labels

                if classifier_key == '3d_cnn':
                    data_for_3d = st.session_state.features if st.session_state.features is not None else st.session_state.preprocessed_data
                    train_info = classifier.fit(
                        train_X, train_y,
                        data_image=data_for_3d,
                        progress_callback=callback
                    )
                elif classifier_key == 'semi_supervised' and use_unlabeled:
                    data_for_semi = st.session_state.features if st.session_state.features is not None else st.session_state.preprocessed_data
                    from src.utils import reshape_for_classifier
                    data_flat = reshape_for_classifier(data_for_semi)
                    n_unlabeled = min(10000, len(data_flat) // 10)
                    indices = np.random.choice(len(data_flat), n_unlabeled, replace=False)
                    X_unlabeled = data_flat[indices]
                    train_info = classifier.fit(
                        train_X, train_y,
                        X_unlabeled=X_unlabeled,
                        progress_callback=callback
                    )
                else:
                    train_info = classifier.fit(
                        train_X, train_y,
                        progress_callback=callback
                    )

                st.session_state.classifier = classifier
                st.session_state.train_info = train_info

                progress_bar.progress(1.0)
                status_text.text("✅ 模型训练完成！")

                st.success("✅ 模型训练完成！")

                if 'best_params' in train_info:
                    st.info(f"🎯 最优参数: {train_info['best_params']}")

                if 'train_losses' in train_info:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(train_info['train_losses']) + 1)),
                        y=train_info['train_losses'],
                        mode='lines',
                        name='训练损失'
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(train_info['train_accuracies']) + 1)),
                        y=train_info['train_accuracies'],
                        mode='lines',
                        name='训练精度',
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        title='训练曲线',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if 'feature_importances' in train_info:
                    importances = train_info['feature_importances']
                    feature_names = st.session_state.feature_info.get('feature_names', [f'Feature {i}' for i in range(len(importances))]) if st.session_state.feature_info else [f'Feature {i}' for i in range(len(importances))]

                    top_idx = np.argsort(importances)[-20:][::-1]
                    fig = px.bar(
                        x=importances[top_idx],
                        y=[feature_names[i] for i in top_idx],
                        orientation='h',
                        title='Top 20 特征重要性'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 模型训练失败: {str(e)}")
            progress_bar.progress(0)
            status_text.text("")
            import traceback
            st.code(traceback.format_exc())

    if st.session_state.classifier is not None and st.session_state.features is not None:
        st.markdown("---")

        if st.button("🔍 执行全图分类", type='primary'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            callback = create_progress_callback(progress_bar, status_text)

            try:
                with st.spinner("正在执行全图分类..."):
                    predictions, _ = classify_image(
                        st.session_state.classifier,
                        st.session_state.features,
                        progress_callback=callback
                    )

                    st.session_state.classification_result = predictions

                    progress_bar.progress(1.0)
                    status_text.text("✅ 全图分类完成！")

                    st.success("✅ 全图分类完成！")

                    class_rgb, legend = classification_to_rgb(
                        predictions,
                        class_names=st.session_state.samples.class_names
                    )
                    st.session_state.classification_rgb = class_rgb
                    st.session_state.classification_legend = legend

            except Exception as e:
                st.error(f"❌ 分类失败: {str(e)}")
                progress_bar.progress(0)
                status_text.text("")


def page_evaluation():
    st.header("📊 精度评估")

    if st.session_state.classifier is None or st.session_state.test_samples is None:
        st.warning("⚠️ 请先完成模型训练和数据集划分")
        return

    if st.button("📐 计算评估指标", type='primary'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        callback = create_progress_callback(progress_bar, status_text)

        try:
            with st.spinner("正在计算评估指标..."):
                eval_result = evaluate_classifier(
                    st.session_state.classifier,
                    st.session_state.test_samples.features,
                    st.session_state.test_samples.labels,
                    class_names=st.session_state.samples.class_names,
                    progress_callback=callback
                )

                st.session_state.metrics = eval_result['metrics']
                st.session_state.y_true = eval_result['y_true']
                st.session_state.y_pred = eval_result['y_pred']

                progress_bar.progress(1.0)
                status_text.text("✅ 评估完成！")

        except Exception as e:
            st.error(f"❌ 评估失败: {str(e)}")
            progress_bar.progress(0)
            status_text.text("")

    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        formatted = format_metrics_for_display(metrics)

        st.subheader("🏆 总体指标")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总体精度 (OA)", formatted['summary']['Overall Accuracy (OA)'])
        col2.metric("平均精度 (AA)", formatted['summary']['Average Accuracy (AA)'])
        col3.metric("Kappa系数", formatted['summary']['Kappa Coefficient'])
        col4.metric("类别数", formatted['summary']['Number of Classes'])

        st.markdown("---")
        st.subheader("📋 各类别详细指标")

        per_class_df = pd.DataFrame(formatted['per_class'])
        st.dataframe(per_class_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🔥 混淆矩阵")

        normalize_cm = st.checkbox("归一化显示", True, key='normalize_cm')

        cm_result = compute_confusion_matrix(
            st.session_state.y_true,
            st.session_state.y_pred,
            normalize=normalize_cm,
            class_names=st.session_state.samples.class_names
        )

        cm_df = pd.DataFrame(
            cm_result['matrix'],
            index=cm_result['tick_labels'],
            columns=cm_result['tick_labels']
        )

        fig = px.imshow(
            cm_df,
            text_auto='.2f' if normalize_cm else 'd',
            color_continuous_scale='Blues',
            title=f"混淆矩阵 {'(归一化)' if normalize_cm else '(原始计数)'}",
            aspect='auto'
        )
        fig.update_layout(
            xaxis_title='预测类别',
            yaxis_title='真实类别',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 类别精度对比")

        classes = list(metrics['per_class_metrics'].keys())
        precisions = [metrics['per_class_metrics'][c]['precision'] for c in classes]
        recalls = [metrics['per_class_metrics'][c]['recall'] for c in classes]
        f1s = [metrics['per_class_metrics'][c]['f1'] for c in classes]
        labels = [metrics['per_class_metrics'][c]['name'] for c in classes]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=precisions, name='精确率'))
        fig.add_trace(go.Bar(x=labels, y=recalls, name='召回率'))
        fig.add_trace(go.Bar(x=labels, y=f1s, name='F1分数'))

        fig.update_layout(
            title='各类别精度指标对比',
            xaxis_title='类别',
            yaxis_title='分数',
            barmode='group',
            yaxis_range=[0, 1.05],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def page_visualization():
    st.header("🖼️ 分类结果可视化")

    if st.session_state.classification_result is None:
        st.warning("⚠️ 请先完成模型训练和分类")
        return

    classification = st.session_state.classification_result
    class_names = st.session_state.samples.class_names
    legend = st.session_state.get('classification_legend', {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎨 分类图显示选项")
        display_mode = st.selectbox(
            "显示模式",
            ["分类图", "叠加显示", "单类提取"],
            key='display_mode'
        )

        if display_mode == "叠加显示":
            background_type = st.selectbox(
                "背景影像",
                ["真彩色", "标准假彩色", "NDVI"],
                key='background_type'
            )
            alpha = st.slider("分类图透明度", 0.1, 1.0, 0.5, 0.05, key='overlay_alpha')

        elif display_mode == "单类提取":
            classes = np.unique(classification)
            target_class = st.selectbox(
                "选择要提取的类别",
                classes,
                format_func=lambda x: class_names.get(x, f'Class {x}'),
                key='target_class'
            )

    with col2:
        st.subheader("🌈 波段组合工具")
        band_combo = st.selectbox(
            "预设组合",
            ["真彩色 (R-G-B)", "标准假彩色 (NIR-R-G)", "NDVI指数", "自定义"],
            key='band_combo'
        )

        if band_combo == "自定义":
            max_band = st.session_state.data.shape[2] - 1
            r_band = st.slider("R通道波段", 0, max_band, min(50, max_band), key='r_band')
            g_band = st.slider("G通道波段", 0, max_band, min(30, max_band), key='g_band')
            b_band = st.slider("B通道波段", 0, max_band, min(10, max_band), key='b_band')

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
            fig = px.pie(
                values=counts,
                names=[class_names.get(u, f'Class {u}') for u in unique],
                title='地物类别分布'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💾 下载结果")

        col1, col2, col3 = st.columns(3)

        with col1:
            from PIL import Image
            class_png = (class_rgb * 255).astype(np.uint8)
            img = Image.fromarray(class_png)
            import io
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            st.download_button(
                "📥 下载分类图 (PNG)",
                buf.getvalue(),
                "classification_map.png",
                "image/png"
            )

        with col2:
            import io
            buf = io.BytesIO()
            np.save(buf, classification)
            st.download_button(
                "📥 下载分类数据 (NPY)",
                buf.getvalue(),
                "classification_result.npy",
                "application/octet-stream"
            )

        with col3:
            if st.session_state.metrics is not None:
                import json
                buf = io.BytesIO()
                metrics_json = json.dumps({
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in st.session_state.metrics.items()
                }, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 下载评估指标 (JSON)",
                    metrics_json,
                    "metrics.json",
                    "application/json"
                )

    except Exception as e:
        st.error(f"❌ 可视化失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def page_batch_processing():
    st.header("📦 批量处理")

    if st.session_state.classifier is None:
        st.warning("⚠️ 请先训练好分类模型再进行批量处理")
        return

    st.markdown("上传多景影像，使用已训练的模型和参数批量执行分类。")

    st.subheader("📁 上传批量数据")

    num_files = st.number_input("要处理的影像数量", 1, 10, 1, key='num_batch_files')

    batch_files = []
    for i in range(num_files):
        st.markdown(f"##### 影像 {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            hdr_file = st.file_uploader(f"头文件 {i+1} (.hdr)", type=['hdr'], key=f'batch_hdr_{i}')
        with col2:
            dat_file = st.file_uploader(f"数据文件 {i+1}", type=[''], key=f'batch_dat_{i}')

        if hdr_file and dat_file:
            batch_files.append((hdr_file, dat_file))

    st.subheader("⚙️ 处理参数")

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.slider("处理块大小", 200, 1000, 500, 50, key='batch_chunk_size')
        overlap = st.slider("块重叠大小", 16, 128, 32, 16, key='batch_overlap')

    with col2:
        output_dir = st.text_input("输出目录", "./batch_outputs", key='output_dir')

    if st.button("🚀 开始批量处理", type='primary'):
        if len(batch_files) < num_files:
            st.warning(f"⚠️ 请上传 {num_files} 组完整的数据文件")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        progress_bar = st.progress(0)
        status_text = st.empty()
        overall_callback = create_progress_callback(progress_bar, status_text)

        try:
            with st.spinner("正在进行批量处理..."):
                saved_files = []
                for hdr_file, dat_file in batch_files:
                    hdr_path = save_uploaded_file(hdr_file, '.hdr')
                    dat_path = save_uploaded_file(dat_file, '')
                    saved_files.append((hdr_path, dat_path))

                jobs = create_batch_jobs(saved_files, output_dir)

                preprocess_steps = []
                feature_config = {
                    'feature_type': 'spectral',
                    'spectral_features': ['continuum_removal', 'first_derivative'],
                    'spatial_features': None
                }

                job_progress = st.progress(0)
                job_status = st.empty()
                job_callback = create_progress_callback(job_progress, job_status)

                results = batch_process(
                    jobs,
                    preprocess_steps,
                    feature_config,
                    st.session_state.classifier,
                    output_dir,
                    wavelengths=st.session_state.wavelengths,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    job_progress_callback=job_callback,
                    overall_progress_callback=overall_callback
                )

                st.success(f"✅ 批量处理完成！")

                success_count = sum(1 for r in results if r.get('success'))
                fail_count = len(results) - success_count

                st.info(f"成功: {success_count} / 失败: {fail_count}")

                if fail_count > 0:
                    for i, r in enumerate(results):
                        if not r.get('success'):
                            st.error(f"影像 {i+1} 失败: {r.get('error', '未知错误')}")

                st.markdown("### 📋 输出文件")
                for i, r in enumerate(results):
                    if r.get('success') and 'output_files' in r:
                        st.markdown(f"**影像 {i+1}:**")
                        for k, v in r['output_files'].items():
                            st.markdown(f"- {k}: `{v}`")

        except Exception as e:
            st.error(f"❌ 批量处理失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def main():
    page = sidebar_navigation()

    if page == '数据导入':
        page_data_import()
    elif page == '数据预处理':
        page_preprocessing()
    elif page == '特征提取':
        page_feature_extraction()
    elif page == '样本管理':
        page_sample_management()
    elif page == '模型训练':
        page_model_training()
    elif page == '精度评估':
        page_evaluation()
    elif page == '分类可视化':
        page_visualization()
    elif page == '批量处理':
        page_batch_processing()

    st.markdown("---")
    st.markdown("🚀 高光谱遥感影像分类与地物识别系统 v1.0")


if __name__ == '__main__':
    main()
