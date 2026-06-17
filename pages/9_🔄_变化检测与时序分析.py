import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import io
import json
from PIL import Image

from state import init_session_state, save_uploaded_file, create_progress_callback
from src.data_io import load_envi_data, get_image_info
from src.visualization import get_true_color, generate_classification_legend
from src.utils import parse_map_info
from src.change_detection import (
    align_images,
    sad_change_detection,
    cva_change_detection,
    pca_change_detection,
    create_change_visualization,
    compute_transition_matrix,
    prepare_sankey_data,
    compute_spectral_difference,
    average_spectrum_in_region,
    export_change_to_geojson,
    run_multi_algorithm_comparison,
    compute_ndvi_pixel_timeseries,
    prepare_chord_diagram_data,
)
from src.classification import classify_image


init_session_state()

st.header("🔄 变化检测与时序分析")
st.info("💡 导入两期高光谱影像，检测地物变化并进行时序分析")


def load_image_phase(uploaded_hdr, uploaded_dat, phase_label):
    if uploaded_hdr is not None:
        hdr_path = save_uploaded_file(uploaded_hdr)
        dat_path = None
        if uploaded_dat is not None:
            dat_path = save_uploaded_file(uploaded_dat)
        try:
            data, header = load_envi_data(hdr_path, dat_path, mmap=True)
            info = get_image_info(data, header)
            st.success(f"✅ {phase_label} 影像加载成功")
            with st.expander(f"📋 {phase_label} 影像信息", expanded=False):
                st.info(
                    f"📐 尺寸: {info['dimensions']}\n\n"
                    f"🎯 波段数: {info['num_bands']}\n\n"
                    f"💾 数据类型: {info['data_type']}\n\n"
                    f"📦 内存占用: {info['memory_mb']:.1f} MB"
                )
            return data, header
        except Exception as e:
            st.error(f"❌ {phase_label} 影像加载失败: {str(e)}")
            return None, None
    return None, None


st.markdown("---")
st.subheader("📥 两期影像导入")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🕐 时相A")
    hdr_a = st.file_uploader("上传头文件 (.hdr)", type=['hdr'], key='hdr_a_upload')
    dat_a = st.file_uploader("上传数据文件 (.dat/.raw)", type=['dat', 'raw', 'img'], key='dat_a_upload')

with col_b:
    st.markdown("### 🕑 时相B")
    hdr_b = st.file_uploader("上传头文件 (.hdr)", type=['hdr'], key='hdr_b_upload')
    dat_b = st.file_uploader("上传数据文件 (.dat/.raw)", type=['dat', 'raw', 'img'], key='dat_b_upload')

col_load_a, col_load_b = st.columns(2)

with col_load_a:
    if st.button("📂 加载时相A影像", key='load_a_btn'):
        data_a, header_a = load_image_phase(hdr_a, dat_a, "时相A")
        if data_a is not None:
            st.session_state.change_data_a = data_a
            st.session_state.change_header_a = header_a
            wavelengths_a = np.array(header_a.wavelengths) if header_a.wavelengths else None
            st.session_state.change_wavelengths_a = wavelengths_a

with col_load_b:
    if st.button("📂 加载时相B影像", key='load_b_btn'):
        data_b, header_b = load_image_phase(hdr_b, dat_b, "时相B")
        if data_b is not None:
            st.session_state.change_data_b = data_b
            st.session_state.change_header_b = header_b
            wavelengths_b = np.array(header_b.wavelengths) if header_b.wavelengths else None
            st.session_state.change_wavelengths_b = wavelengths_b


def check_and_align_images():
    if st.session_state.change_data_a is None or st.session_state.change_data_b is None:
        st.warning("⚠️ 请先加载两期影像")
        return False

    data_a = st.session_state.change_data_a
    data_b = st.session_state.change_data_b

    Ha, Wa, Ba = data_a.shape
    Hb, Wb, Bb = data_b.shape

    size_match = (Ha == Hb) and (Wa == Wb) and (Ba == Bb)

    if size_match:
        st.success("✅ 两期影像尺寸和波段数完全一致")
        st.session_state.change_aligned_a = data_a
        st.session_state.change_aligned_b = data_b
        st.session_state.change_align_info = {
            'original_shape_a': (Ha, Wa, Ba),
            'original_shape_b': (Hb, Wb, Bb),
            'bands_match': True,
            'spatial_match': True,
            'aligned_shape': (Ha, Wa, Ba),
            'align_mode': 'direct',
        }
        return True
    else:
        st.warning("⚠️ 两期影像尺寸或波段数不一致，需要进行对齐处理")
        st.info(
            f"时相A: {Ha} × {Wa} × {Ba}\n\n"
            f"时相B: {Hb} × {Wb} × {Bb}"
        )

        align_mode = st.selectbox(
            "选择对齐方式",
            ["左上角裁剪 (crop_min)", "中心裁剪 (center_crop)"],
            index=0,
            key='align_mode_select'
        )

        if st.button("🔧 执行对齐", key='do_align_btn'):
            mode = 'crop_min' if '左上角' in align_mode else 'center_crop'
            try:
                aligned_a, aligned_b, info = align_images(data_a, data_b, align_mode=mode)
                st.session_state.change_aligned_a = aligned_a
                st.session_state.change_aligned_b = aligned_b
                st.session_state.change_align_info = info

                H, W, B = info['aligned_shape']
                st.success(f"✅ 对齐完成，对齐后尺寸: {H} × {W} × {B}")
                return True
            except Exception as e:
                st.error(f"❌ 对齐失败: {str(e)}")
                return False

    return False


st.markdown("---")
st.subheader("⚙️ 影像对齐")

images_aligned = False

if st.session_state.change_aligned_a is not None and st.session_state.change_aligned_b is not None:
    info = st.session_state.change_align_info
    H, W, B = info['aligned_shape']
    st.success(f"✅ 影像已对齐，尺寸: {H} × {W} × {B}")
    images_aligned = True
else:
    images_aligned = check_and_align_images()


if images_aligned and st.session_state.change_aligned_a is not None:
    st.markdown("---")
    st.subheader("🔬 真彩色预览")

    try:
        rgb_a = get_true_color(
            st.session_state.change_aligned_a,
            st.session_state.change_wavelengths_a
        )
        rgb_b = get_true_color(
            st.session_state.change_aligned_b,
            st.session_state.change_wavelengths_b
        )

        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            fig_a, ax_a = plt.subplots(figsize=(8, 6))
            ax_a.imshow(rgb_a)
            ax_a.set_title('时相A 真彩色合成')
            ax_a.axis('off')
            st.pyplot(fig_a)

        with prev_col2:
            fig_b, ax_b = plt.subplots(figsize=(8, 6))
            ax_b.imshow(rgb_b)
            ax_b.set_title('时相B 真彩色合成')
            ax_b.axis('off')
            st.pyplot(fig_b)
    except Exception as e:
        st.error(f"❌ 预览生成失败: {str(e)}")


    st.markdown("---")
    st.subheader("📊 变化检测算法")

    method = st.selectbox(
        "选择变化检测算法",
        ["光谱角距离法 (SAD)", "变化向量分析法 (CVA)", "基于PCA的变化检测"],
        index=0,
        key='cd_method_select'
    )

    st.markdown("#### ⚙️ 算法参数")

    params_ready = True
    method_key = ''

    if method == "光谱角距离法 (SAD)":
        method_key = 'SAD'
        sad_threshold = st.slider(
            "光谱角阈值 (弧度)",
            0.01, 1.0, 0.1, 0.01,
            key='sad_threshold_slider'
        )
        st.caption("💡 角度越大表示光谱差异越大，一般取 0.05-0.3 弧度")

    elif method == "变化向量分析法 (CVA)":
        method_key = 'CVA'
        cva_threshold_method = st.selectbox(
            "阈值确定方法",
            ["手动设置", "百分位数法", "均值+2倍标准差"],
            index=1,
            key='cva_threshold_method'
        )

        if cva_threshold_method == "手动设置":
            cva_threshold = st.number_input(
                "变化向量模阈值",
                0.0, 10000.0, 100.0, 1.0,
                key='cva_threshold_input'
            )
        elif cva_threshold_method == "百分位数法":
            cva_percentile = st.slider(
                "变化像素占比 (%)",
                1.0, 50.0, 5.0, 0.5,
                key='cva_percentile_slider'
            )
        else:
            cva_threshold = None

    else:
        method_key = 'PCA'
        pca_variance_ratio = st.slider(
            "累积方差贡献率",
            0.5, 0.99, 0.95, 0.01,
            key='pca_variance_slider'
        )
        st.caption("💡 取前N个主成分，使累积方差贡献达到设定比例")

    if st.button("🚀 执行变化检测", key='run_cd_btn'):
        with st.spinner("正在进行变化检测..."):
            try:
                img_a = st.session_state.change_aligned_a
                img_b = st.session_state.change_aligned_b
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_cb(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status_text.text(m)

                if method_key == 'SAD':
                    change_mask, intensity, stats = sad_change_detection(
                        img_a, img_b,
                        threshold=sad_threshold,
                        chunk_size=500,
                        progress_callback=progress_cb
                    )
                elif method_key == 'CVA':
                    if cva_threshold_method == "手动设置":
                        change_mask, intensity, stats = cva_change_detection(
                            img_a, img_b,
                            threshold=cva_threshold,
                            threshold_method='manual',
                            chunk_size=500,
                            progress_callback=progress_cb
                        )
                    elif cva_threshold_method == "百分位数法":
                        change_mask, intensity, stats = cva_change_detection(
                            img_a, img_b,
                            threshold=None,
                            threshold_method='percentile',
                            percentile=100 - cva_percentile,
                            chunk_size=500,
                            progress_callback=progress_cb
                        )
                    else:
                        change_mask, intensity, stats = cva_change_detection(
                            img_a, img_b,
                            threshold=None,
                            threshold_method='auto',
                            chunk_size=500,
                            progress_callback=progress_cb
                        )
                else:
                    change_mask, intensity, stats = pca_change_detection(
                        img_a, img_b,
                        variance_ratio=pca_variance_ratio,
                        chunk_size=500,
                        progress_callback=progress_cb
                    )

                st.session_state.change_method = method_key
                st.session_state.change_mask = change_mask
                st.session_state.change_intensity = intensity
                st.session_state.change_stats = stats

                st.success("✅ 变化检测完成")

            except Exception as e:
                st.error(f"❌ 变化检测失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


    if st.session_state.change_mask is not None:
        st.markdown("---")
        st.subheader("📈 变化检测结果")

        stats = st.session_state.change_stats

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("总像素数", f"{stats['total_pixels']:,}")
        col_stat2.metric("变化像素数", f"{stats['change_pixels']:,}")
        col_stat3.metric("未变化像素数", f"{stats['total_pixels'] - stats['change_pixels']:,}")
        col_stat4.metric("变化面积占比", f"{stats['change_ratio']*100:.2f}%")

        with st.expander("📋 详细统计信息", expanded=False):
            if stats['method'] == 'SAD':
                st.info(
                    f"📊 SAD统计:\n\n"
                    f"• 平均角度: {stats['sad_mean']:.4f} rad\n\n"
                    f"• 标准差: {stats['sad_std']:.4f} rad\n\n"
                    f"• 最小值: {stats['sad_min']:.4f} rad\n\n"
                    f"• 最大值: {stats['sad_max']:.4f} rad\n\n"
                    f"• 阈值: {stats['threshold']:.4f} rad"
                )
            elif stats['method'] == 'CVA':
                st.info(
                    f"📊 CVA统计:\n\n"
                    f"• 平均模值: {stats['magnitude_mean']:.4f}\n\n"
                    f"• 标准差: {stats['magnitude_std']:.4f}\n\n"
                    f"• 最小值: {stats['magnitude_min']:.4f}\n\n"
                    f"• 最大值: {stats['magnitude_max']:.4f}\n\n"
                    f"• 阈值: {stats['threshold']:.4f}"
                )
            else:
                st.info(
                    f"📊 PCA统计:\n\n"
                    f"• 主成分数: {stats['n_components']}\n\n"
                    f"• 累积方差贡献率: {stats['cumulative_variance']*100:.2f}%\n\n"
                    f"• OTSU阈值: {stats['threshold']:.4f}\n\n"
                    f"• 平均模值: {stats['magnitude_mean']:.4f}\n\n"
                    f"• 标准差: {stats['magnitude_std']:.4f}"
                )

        try:
            bg_rgb = get_true_color(
                st.session_state.change_aligned_a,
                st.session_state.change_wavelengths_a
            )

            binary_vis, heat_vis = create_change_visualization(
                st.session_state.change_mask,
                st.session_state.change_intensity,
                background=bg_rgb
            )

            st.session_state.change_binary_vis = binary_vis
            st.session_state.change_heat_vis = heat_vis

            col_geo1, col_geo2 = st.columns([1, 1])
            with col_geo1:
                min_area_pixels = st.number_input(
                    "最小区域面积（像素）",
                    1, 10000, 10, 1,
                    key='geojson_min_area'
                )
            with col_geo2:
                pixel_size = st.number_input(
                    "像素大小（米，可选）",
                    0.0, 10000.0, 0.0, 0.1,
                    key='geojson_pixel_size'
                )

            header_a = st.session_state.change_header_a
            map_info = None
            coord_sys = None
            if header_a and header_a.map_info:
                map_info = parse_map_info(header_a.map_info)
                coord_sys = header_a.coordinate_system_string if header_a.coordinate_system_string else None
                if map_info:
                    st.caption(f"🗺️ 检测到地理参考: {map_info['projection']}, 像素大小: {map_info['pixel_size'][0]:.4f} × {map_info['pixel_size'][1]:.4f} {map_info.get('units', 'Meters')}")
                else:
                    st.caption("⚠️ 头文件中有map_info但解析失败，将使用像素坐标")
            else:
                st.caption("⚠️ 影像无头文件地理参考信息，导出坐标为像素行列号")

            if st.button("🗺️ 生成GeoJSON矢量结果", key='gen_geojson_btn'):
                with st.spinner("正在提取变化区域并生成GeoJSON..."):
                    try:
                        geojson = export_change_to_geojson(
                            st.session_state.change_mask,
                            st.session_state.change_intensity,
                            class_a=st.session_state.change_class_a,
                            class_b=st.session_state.change_class_b,
                            class_names=st.session_state.samples.class_names if st.session_state.samples else None,
                            min_area_pixels=min_area_pixels,
                            pixel_size=pixel_size if pixel_size > 0 else None,
                            map_info=map_info,
                            coordinate_system=coord_sys,
                        )
                        st.session_state.change_geojson = geojson
                        n_features = len(geojson['features'])
                        coord_type = geojson['metadata']['coordinate_type']
                        coord_label = '地理坐标' if coord_type == 'geographic' else '像素坐标'
                        st.success(f"✅ GeoJSON生成成功，共提取 {n_features} 个变化区域")
                        st.info(f"📊 元数据: 总变化像素 {geojson['metadata']['total_change_pixels']:,}, "
                                f"变化占比 {geojson['metadata']['change_ratio']*100:.2f}%, "
                                f"坐标类型: {coord_label}")
                    except Exception as e:
                        st.error(f"❌ GeoJSON生成失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            st.markdown("#### 🖼️ 结果可视化")

            view_col1, view_col2, view_col3 = st.columns(3)

            with view_col1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.imshow(rgb_a)
                ax1.set_title('时相A 真彩色')
                ax1.axis('off')
                st.pyplot(fig1)

            with view_col2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.imshow(binary_vis)
                ax2.set_title('变化检测二值图\n(红色=变化, 灰色=未变化)')
                ax2.axis('off')
                st.pyplot(fig2)

            with view_col3:
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                ax3.imshow(rgb_b)
                ax3.set_title('时相B 真彩色')
                ax3.axis('off')
                st.pyplot(fig3)

            st.markdown("##### 🔥 变化强度热力图")
            fig_h, ax_h = plt.subplots(figsize=(12, 8))
            im = ax_h.imshow(st.session_state.change_intensity, cmap='hot')
            ax_h.set_title('变化强度热力图 (颜色越深变化越大)')
            ax_h.axis('off')
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            st.pyplot(fig_h)

            st.markdown("##### 📊 变化强度直方图")
            intensity_flat = st.session_state.change_intensity.ravel()
            fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
            ax_hist.hist(intensity_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='white')
            if stats['method'] == 'SAD':
                threshold = stats['threshold']
                ax_hist.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                               label=f'阈值 = {threshold:.4f}')
            elif stats['method'] == 'CVA':
                threshold = stats['threshold']
                ax_hist.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                               label=f'阈值 = {threshold:.4f}')
            else:
                threshold = stats['threshold']
                ax_hist.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                               label=f'OTSU阈值 = {threshold:.4f}')
            ax_hist.set_xlabel('变化强度')
            ax_hist.set_ylabel('像素数')
            ax_hist.set_title('变化强度分布直方图')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)

        except Exception as e:
            st.error(f"❌ 可视化失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


        st.markdown("---")
        st.subheader("⚖️ 多算法对比分析")

        st.info("💡 选择2-3种变化检测算法进行对比，分析各算法的一致性和差异")

        col_ma1, col_ma2 = st.columns([2, 1])
        with col_ma1:
            selected_algos = st.multiselect(
                "选择要对比的算法（至少2种，最多3种）",
                ["SAD - 光谱角距离法", "CVA - 变化向量分析法", "PCA - 主成分分析法"],
                default=["SAD - 光谱角距离法", "CVA - 变化向量分析法"],
                key='multi_algo_select'
            )
        with col_ma2:
            st.markdown("#### ")
            run_multi = st.button("🚀 执行多算法对比", key='run_multi_algo_btn')

        if len(selected_algos) > 0:
            with st.expander("⚙️ 各算法参数设置", expanded=False):
                algo_params = {}
                if "SAD - 光谱角距离法" in selected_algos:
                    st.markdown("**SAD 参数**")
                    sad_th = st.slider("光谱角阈值 (弧度)", 0.01, 1.0, 0.1, 0.01, key='ma_sad_th')
                    algo_params['SAD'] = {'threshold': sad_th}

                if "CVA - 变化向量分析法" in selected_algos:
                    st.markdown("**CVA 参数**")
                    cva_method = st.selectbox(
                        "阈值确定方法",
                        ["百分位数法", "手动设置", "均值+2倍标准差"],
                        index=0, key='ma_cva_method'
                    )
                    cva_params = {}
                    if cva_method == "手动设置":
                        cva_params['threshold'] = st.number_input("变化向量模阈值", 0.0, 10000.0, 100.0, 1.0, key='ma_cva_th')
                        cva_params['threshold_method'] = 'manual'
                    elif cva_method == "百分位数法":
                        cva_pct = st.slider("变化像素占比 (%)", 1.0, 50.0, 5.0, 0.5, key='ma_cva_pct')
                        cva_params['threshold_method'] = 'percentile'
                        cva_params['percentile'] = 100 - cva_pct
                    else:
                        cva_params['threshold_method'] = 'auto'
                    algo_params['CVA'] = cva_params

                if "PCA - 主成分分析法" in selected_algos:
                    st.markdown("**PCA 参数**")
                    pca_var = st.slider("累积方差贡献率", 0.5, 0.99, 0.95, 0.01, key='ma_pca_var')
                    algo_params['PCA'] = {'variance_ratio': pca_var}

        if run_multi:
            algo_keys = []
            for a in selected_algos:
                if a.startswith('SAD'):
                    algo_keys.append('SAD')
                elif a.startswith('CVA'):
                    algo_keys.append('CVA')
                elif a.startswith('PCA'):
                    algo_keys.append('PCA')

            if len(algo_keys) < 2:
                st.warning("⚠️ 请至少选择2种算法进行对比")
            else:
                with st.spinner("正在并行运行多算法对比..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def pcb(p, m):
                            progress_bar.progress(min(p, 1.0))
                            status_text.text(m)

                        comparison = run_multi_algorithm_comparison(
                            st.session_state.change_aligned_a,
                            st.session_state.change_aligned_b,
                            algorithms=algo_keys,
                            params=algo_params,
                            chunk_size=500,
                            progress_callback=pcb
                        )
                        st.session_state.multi_algo_comparison = comparison
                        st.success("✅ 多算法对比完成")
                    except Exception as e:
                        st.error(f"❌ 多算法对比失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        if st.session_state.multi_algo_comparison is not None:
            comp = st.session_state.multi_algo_comparison

            st.markdown("#### 📊 算法检测结果对比")

            algo_names_map = {'SAD': '光谱角距离法(SAD)', 'CVA': '变化向量分析法(CVA)', 'PCA': '主成分分析法(PCA)'}

            col_ar1, col_ar2, col_ar3 = st.columns(3)
            area_ratios = comp['area_ratios']
            algo_cols = [col_ar1, col_ar2, col_ar3]
            for idx, (algo, ratio) in enumerate(area_ratios.items()):
                if idx < len(algo_cols):
                    algo_cols[idx].metric(
                        f"{algo_names_map.get(algo, algo)}",
                        f"{ratio*100:.3f}%",
                        delta=f"变化面积占比"
                    )

            st.markdown("#### 🎯 Kappa一致性系数矩阵")

            algo_list = list(area_ratios.keys())
            n_algos = len(algo_list)
            algo_labels = [algo_names_map.get(a, a) for a in algo_list]

            kappa_matrix = np.ones((n_algos, n_algos))
            hover_text = []

            for i in range(n_algos):
                row_hover = []
                for j in range(n_algos):
                    if i == j:
                        kappa_matrix[i, j] = 1.0
                        row_hover.append(f"<b>{algo_labels[i]} vs {algo_labels[j]}</b><br>Kappa: 1.0000<br><i>自身对比，完全一致</i>")
                    else:
                        key = (algo_list[i], algo_list[j]) if (algo_list[i], algo_list[j]) in comp['kappa_matrix'] else (algo_list[j], algo_list[i])
                        kappa_val = comp['kappa_matrix'].get(key, 0.0)
                        kappa_matrix[i, j] = round(kappa_val, 4)

                        if kappa_val < 0:
                            level = "无一致性"
                        elif kappa_val < 0.2:
                            level = "极弱一致性"
                        elif kappa_val < 0.4:
                            level = "弱一致性"
                        elif kappa_val < 0.6:
                            level = "中等一致性"
                        elif kappa_val < 0.8:
                            level = "强一致性"
                        else:
                            level = "极强一致性"

                        row_hover.append(
                            f"<b>{algo_labels[i]} vs {algo_labels[j]}</b><br>"
                            f"Kappa系数: {kappa_val:.4f}<br>"
                            f"评价: {level}"
                        )
                hover_text.append(row_hover)

            fig_kappa = go.Figure(data=go.Heatmap(
                z=kappa_matrix,
                x=algo_labels,
                y=algo_labels,
                text=kappa_matrix,
                texttemplate='%{text:.4f}',
                hovertemplate='%{hovertext}<extra></extra>',
                customdata=hover_text,
                hovertext=hover_text,
                colorscale='RdYlGn',
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(
                    title='Kappa系数',
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['0<br>无', '0.2<br>极弱', '0.4<br>弱', '0.6<br>中等', '0.8<br>强', '1.0<br>极强'],
                    thickness=20,
                ),
            ))

            fig_kappa.update_layout(
                title='Kappa一致性系数矩阵',
                xaxis_title='算法',
                yaxis_title='算法',
                height=400 + n_algos * 30,
                margin=dict(l=10, r=10, t=50, b=10),
            )

            fig_kappa.update_yaxes(autorange='reversed')

            st.plotly_chart(fig_kappa, use_container_width=True)

            st.caption("💡 Kappa系数评价标准: <0 无一致性, 0-0.2 极弱, 0.2-0.4 弱, 0.4-0.6 中等, 0.6-0.8 强, 0.8-1 极强。对角线为自身对比，Kappa恒为1.0")

            st.markdown("#### 🗺️ 变化区域重叠度热力图")

            overlap = comp['overlap_heatmap']
            n_algos = len(algo_list)

            fig_overlap, ax_overlap = plt.subplots(figsize=(10, 8))
            cmap_overlap = plt.cm.get_cmap('YlOrRd', n_algos + 1)
            bounds = np.arange(-0.5, n_algos + 1.5, 1)
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap_overlap.N)
            im_ov = ax_overlap.imshow(overlap, cmap=cmap_overlap, norm=norm, interpolation='nearest')
            ax_overlap.set_title(f'变化区域重叠度 (共{n_algos}种算法)')
            ax_overlap.axis('off')
            cbar = plt.colorbar(im_ov, ax=ax_overlap, ticks=np.arange(0, n_algos + 1), fraction=0.046, pad=0.04)
            cbar.set_ticklabels([f'{i}种算法检测到变化' for i in range(n_algos + 1)])
            st.pyplot(fig_overlap)

            total_pixels = overlap.size
            for n in range(n_algos + 1):
                count = int(np.sum(overlap == n))
                pct = count / total_pixels * 100
                if n == 0:
                    st.info(f"⚪ 无算法检测为变化: {count:,} 像素 ({pct:.2f}%)")
                elif n == n_algos:
                    st.success(f"🟢 所有{n_algos}种算法一致检测为变化: {count:,} 像素 ({pct:.2f}%)")
                else:
                    st.warning(f"🟡 仅{n}种算法检测为变化: {count:,} 像素 ({pct:.2f}%)")

            st.markdown("#### 🖼️ 各算法检测结果对比")

            n_results = len(comp['algorithm_results'])
            result_cols = st.columns(n_results)
            for idx, (algo, res) in enumerate(comp['algorithm_results'].items()):
                with result_cols[idx]:
                    bin_v, _ = create_change_visualization(
                        res['mask'], res['intensity']
                    )
                    fig_r, ax_r = plt.subplots(figsize=(6, 5))
                    ax_r.imshow(bin_v)
                    ax_r.set_title(f'{algo_names_map.get(algo, algo)}\n变化像素: {res["stats"]["change_pixels"]:,}')
                    ax_r.axis('off')
                    st.pyplot(fig_r)


        st.markdown("---")
        st.subheader("🔄 变化区域分类与转移分析")

        st.info("💡 使用已训练的分类器对两期影像进行分类，分析地物类别转移")

        if st.session_state.classifier is None:
            st.warning("⚠️ 尚未训练分类器。请先在「模型训练」页面训练分类器。")
        else:
            if st.button("🔍 对变化区域进行分类分析", key='classify_change_btn'):
                with st.spinner("正在进行分类分析..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def cb(p, m):
                            progress_bar.progress(min(p, 1.0))
                            status_text.text(m)

                        cb(0.1, "分类时相A影像...")
                        class_a_result, _ = classify_image(
                            st.session_state.classifier,
                            st.session_state.change_aligned_a,
                            chunk_size=500
                        )

                        cb(0.5, "分类时相B影像...")
                        class_b_result, _ = classify_image(
                            st.session_state.classifier,
                            st.session_state.change_aligned_b,
                            chunk_size=500
                        )

                        cb(0.7, "计算转移矩阵...")
                        class_names = st.session_state.samples.class_names
                        transition_matrix, classes, trans_stats = compute_transition_matrix(
                            class_a_result,
                            class_b_result,
                            class_names,
                            change_mask=st.session_state.change_mask
                        )

                        cb(0.9, "准备桑基图数据...")
                        sankey = prepare_sankey_data(
                            transition_matrix,
                            classes,
                            class_names
                        )

                        st.session_state.change_class_a = class_a_result
                        st.session_state.change_class_b = class_b_result
                        st.session_state.transition_matrix = transition_matrix
                        st.session_state.transition_classes = classes
                        st.session_state.transition_stats = trans_stats
                        st.session_state.sankey_data = sankey

                        cb(1.0, "分类分析完成")
                        st.success("✅ 变化区域分类分析完成")

                    except Exception as e:
                        st.error(f"❌ 分类分析失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


        if st.session_state.transition_matrix is not None:
            st.markdown("#### 📊 转移矩阵")

            tm = st.session_state.transition_matrix
            classes = st.session_state.transition_classes
            class_names = st.session_state.samples.class_names

            class_label_list = [class_names.get(c, f'Class {c}') for c in classes]

            tm_df = pd.DataFrame(
                tm,
                index=[f'{name} (A)' for name in class_label_list],
                columns=[f'{name} (B)' for name in class_label_list]
            )

            st.dataframe(tm_df, use_container_width=True)

            n_classes = len(classes)

            if n_classes > 6:
                st.markdown("#### 🎻 弦图 - 类别转移流向 (类别较多自动切换)")
                st.caption(f"💡 当前类别数: {n_classes}，超过6个类别自动使用弦图展示")
                try:
                    chord_data = prepare_chord_diagram_data(tm, classes, class_names)
                    st.session_state.chord_data = chord_data

                    labels = chord_data['labels']
                    matrix = chord_data['matrix']
                    n = chord_data['n_classes']
                    total_pixels = chord_data['total_pixels']

                    color_list = px.colors.qualitative.Plotly
                    if n > len(color_list):
                        color_list = color_list * ((n // len(color_list)) + 1)
                    node_colors = color_list[:n]

                    fig_chord = go.Figure()

                    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    segment_half_width = 0.05 * np.pi / n
                    arc_radius = 0.9
                    inner_radius = 0.6

                    for i in range(n):
                        start_angle = theta[i] - segment_half_width * (2 * np.pi / n) * n / 10
                        end_angle = theta[i] + segment_half_width * (2 * np.pi / n) * n / 10

                        total_i = sum(matrix[i]) + sum(row[i] for row in matrix)
                        if total_i == 0:
                            total_i = 1

                        n_pts = 50
                        angles_outer = np.linspace(start_angle, end_angle, n_pts)
                        angles_inner = np.linspace(end_angle, start_angle, n_pts)

                        x_outer = arc_radius * np.cos(angles_outer)
                        y_outer = arc_radius * np.sin(angles_outer)
                        x_inner = (arc_radius - 0.05) * np.cos(angles_inner)
                        y_inner = (arc_radius - 0.05) * np.sin(angles_inner)

                        x_path = np.concatenate([x_outer, x_inner, [x_outer[0]]])
                        y_path = np.concatenate([y_outer, y_inner, [y_outer[0]]])

                        hover_text = (
                            f"<b>{labels[i]}</b><br>"
                            f"总关联像素: {int(total_i):,}<br>"
                            f"占比: {total_i/total_pixels*100:.2f}%" if total_pixels > 0 else ""
                        )

                        fig_chord.add_trace(go.Scatter(
                            x=x_path,
                            y=y_path,
                            fill='toself',
                            fillcolor=node_colors[i],
                            line=dict(width=0.5, color='white'),
                            mode='lines',
                            text=hover_text,
                            hoverinfo='text',
                            showlegend=True,
                            name=labels[i],
                            legendgroup=f'node_{i}',
                        ))

                        label_radius = arc_radius + 0.12
                        label_x = label_radius * np.cos(theta[i])
                        label_y = label_radius * np.sin(theta[i])

                        angle_deg = theta[i] * 180 / np.pi

                        if -30 <= angle_deg <= 30:
                            xanchor = 'left'
                            yanchor = 'middle'
                        elif 30 < angle_deg <= 60:
                            xanchor = 'left'
                            yanchor = 'bottom'
                        elif 60 < angle_deg <= 120:
                            xanchor = 'center'
                            yanchor = 'bottom'
                        elif 120 < angle_deg <= 150:
                            xanchor = 'right'
                            yanchor = 'bottom'
                        elif 150 < angle_deg <= 210:
                            xanchor = 'right'
                            yanchor = 'middle'
                        elif 210 < angle_deg <= 240:
                            xanchor = 'right'
                            yanchor = 'top'
                        elif 240 < angle_deg <= 300:
                            xanchor = 'center'
                            yanchor = 'top'
                        elif 300 < angle_deg <= 330:
                            xanchor = 'left'
                            yanchor = 'top'
                        else:
                            xanchor = 'left'
                            yanchor = 'middle'

                        fig_chord.add_annotation(
                            x=label_x,
                            y=label_y,
                            text=labels[i],
                            showarrow=False,
                            font=dict(size=9, color='#333333'),
                            xanchor=xanchor,
                            yanchor=yanchor,
                            bgcolor='rgba(255,255,255,0.85)',
                            borderpad=2,
                        )

                    max_val = max([matrix[i][j] for i in range(n) for j in range(n) if i != j] + [1])
                    for i in range(n):
                        for j in range(n):
                            if i != j and matrix[i][j] > 0:
                                value = matrix[i][j]
                                width = 0.005 + 0.05 * (value / max_val)

                                t = np.linspace(0, 1, 50)
                                angle_i = theta[i]
                                angle_j = theta[j]

                                mid_angle = (angle_i + angle_j) / 2
                                angle_diff = abs(angle_j - angle_i)
                                if angle_diff > np.pi:
                                    mid_angle += np.pi

                                ctrl_radius = inner_radius * 0.5
                                x_start = inner_radius * np.cos(angle_i)
                                y_start = inner_radius * np.sin(angle_i)
                                x_end = inner_radius * np.cos(angle_j)
                                y_end = inner_radius * np.sin(angle_j)
                                x_mid = ctrl_radius * np.cos(mid_angle)
                                y_mid = ctrl_radius * np.sin(mid_angle)

                                bezier_x = (1-t)**2 * x_start + 2*(1-t)*t * x_mid + t**2 * x_end
                                bezier_y = (1-t)**2 * y_start + 2*(1-t)*t * y_mid + t**2 * y_end

                                detail = next(
                                    (d for d in chord_data['details']
                                     if d['source_idx'] == i and d['target_idx'] == j),
                                    None
                                )
                                pct = f"{detail['percentage']:.2f}%" if detail else "0.00%"

                                color_i = node_colors[i]
                                hover_text = (
                                    f"<b>{labels[i]} → {labels[j]}</b><br>"
                                    f"转移像素数: {int(value):,}<br>"
                                    f"占比: {pct}"
                                )

                                fig_chord.add_trace(go.Scatter(
                                    x=bezier_x,
                                    y=bezier_y,
                                    mode='lines',
                                    line=dict(width=width * 100, color=color_i),
                                    opacity=0.5,
                                    text=hover_text,
                                    hoverinfo='text',
                                    showlegend=False,
                                    hoverlabel=dict(bgcolor='white'),
                                ))

                    fig_chord.update_layout(
                        title_text=f'地物类别转移弦图 ({n}个类别)',
                        showlegend=True,
                        height=600 + max(0, n - 8) * 30,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                   scaleanchor="y", scaleratio=1, range=[-1.5, 1.5]),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=9),
                            itemwidth=30,
                        ),
                    )

                    st.plotly_chart(fig_chord, use_container_width=True)

                    with st.expander("📋 详细转移信息", expanded=False):
                        detail_df = pd.DataFrame(chord_data['details'])
                        detail_df = detail_df.sort_values('pixels', ascending=False).reset_index(drop=True)
                        detail_df_display = detail_df[['source', 'target', 'pixels', 'percentage']].copy()
                        detail_df_display.columns = ['时相A类别', '时相B类别', '转移像素数', '占比(%)']
                        detail_df_display['占比(%)'] = detail_df_display['占比(%)'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(detail_df_display, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 弦图绘制失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            else:
                st.markdown("#### 🌊 桑基图 - 类别转移流向")

                try:
                    sankey = st.session_state.sankey_data

                    fig_sankey = go.Figure(data=[go.Sankey(
                        valueformat=",d",
                        valuesuffix=" 像素",
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=sankey['labels'],
                            color=px.colors.qualitative.Plotly[:len(sankey['labels'])],
                            hovertemplate='%{label}<br>像素数: %{value:,}<extra></extra>'
                        ),
                        link=dict(
                            source=sankey['source'],
                            target=sankey['target'],
                            value=sankey['value'],
                            hovertemplate='从 %{source.label} 到 %{target.label}<br>'
                                          '转移像素数: %{value:,}<br>'
                                          '占比: %{percent:.2%}<extra></extra>'
                        )
                    )])

                    fig_sankey.update_layout(
                        title_text="地物类别转移桑基图",
                        font_size=12,
                        height=600,
                    )

                    st.plotly_chart(fig_sankey, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 桑基图绘制失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            st.markdown("#### 📈 主要转移类型统计")

            try:
                n = len(classes)
                transitions = []
                total_changed = int(np.sum(tm))

                for i in range(n):
                    for j in range(n):
                        if i != j and tm[i, j] > 0:
                            name_a = class_names.get(classes[i], f'Class {classes[i]}')
                            name_b = class_names.get(classes[j], f'Class {classes[j]}')
                            count = int(tm[i, j])
                            transitions.append({
                                '时相A类别': name_a,
                                '时相B类别': name_b,
                                '转移像素数': count,
                                '占变化像素比例': f"{count/total_changed*100:.2f}%"
                            })

                transitions_df = pd.DataFrame(transitions)
                transitions_df = transitions_df.sort_values('转移像素数', ascending=False).reset_index(drop=True)
                st.dataframe(transitions_df.head(20), use_container_width=True)

                if len(transitions_df) > 0:
                    top_n = min(10, len(transitions_df))
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=[f"{row['时相A类别']}→{row['时相B类别']}" for _, row in transitions_df.head(top_n).iterrows()],
                        y=transitions_df.head(top_n)['转移像素数'].tolist(),
                        text=[f"{v:,}" for v in transitions_df.head(top_n)['转移像素数'].tolist()],
                        textposition='auto',
                        marker_color=px.colors.sequential.RdBu_r[:top_n]
                    ))
                    fig_bar.update_layout(
                        title=f'Top {top_n} 地物转移类型',
                        xaxis_title='转移类型',
                        yaxis_title='转移像素数',
                        height=500,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 转移统计失败: {str(e)}")


        st.markdown("---")
        st.subheader("📈 时序光谱曲线分析")

        temporal_mode = st.radio(
            "时序分析模式",
            ["两期对比（传统模式）", "多期时序分析（3期及以上）"],
            horizontal=True,
            key='temporal_mode_radio'
        )

        if temporal_mode == "两期对比（传统模式）":
            st.info("💡 点击图像或框选区域，查看两期光谱曲线变化对比")

            analysis_mode = st.radio(
                "选择分析模式",
                ["单像素光谱对比", "区域平均光谱对比"],
                horizontal=True,
                key='spectral_analysis_mode'
            )

            if analysis_mode == "单像素光谱对比":
                st.caption("💡 输入像素坐标查看光谱曲线对比")

                col_px, col_py = st.columns(2)
                H, W = st.session_state.change_aligned_a.shape[:2]

                with col_px:
                    px_x = st.number_input("像素 X 坐标 (列)", 0, W-1, W//2, 1, key='px_x_input')
                with col_py:
                    px_y = st.number_input("像素 Y 坐标 (行)", 0, H-1, H//2, 1, key='px_y_input')

                if st.button("🔍 查看该像素光谱", key='view_pixel_spectrum_btn'):
                    try:
                        spec_a = st.session_state.change_aligned_a[px_y, px_x, :].astype(np.float64)
                        spec_b = st.session_state.change_aligned_b[px_y, px_x, :].astype(np.float64)

                        wavelengths = st.session_state.change_wavelengths_a
                        if wavelengths is None:
                            wavelengths = np.arange(len(spec_a))

                        diff_stats = compute_spectral_difference(spec_a, spec_b, wavelengths)

                        st.session_state.change_selected_pixel = {
                            'x': px_x,
                            'y': px_y,
                            'spectrum_a': spec_a,
                            'spectrum_b': spec_b,
                            'wavelengths': wavelengths,
                            'diff_stats': diff_stats,
                        }

                    except Exception as e:
                        st.error(f"❌ 读取像素光谱失败: {str(e)}")

            else:
                st.caption("💡 输入区域坐标范围查看平均光谱")

                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                H, W = st.session_state.change_aligned_a.shape[:2]

                with col_r1:
                    reg_x1 = st.number_input("起始 X", 0, W-1, W//4, 1, key='reg_x1')
                with col_r2:
                    reg_x2 = st.number_input("结束 X", 0, W-1, W//2, 1, key='reg_x2')
                with col_r3:
                    reg_y1 = st.number_input("起始 Y", 0, H-1, H//4, 1, key='reg_y1')
                with col_r4:
                    reg_y2 = st.number_input("结束 Y", 0, H-1, H//2, 1, key='reg_y2')

                if st.button("🔍 查看区域平均光谱", key='view_region_spectrum_btn'):
                    try:
                        x1, x2 = min(reg_x1, reg_x2), max(reg_x1, reg_x2)
                        y1, y2 = min(reg_y1, reg_y2), max(reg_y1, reg_y2)

                        region_mask = np.zeros((H, W), dtype=bool)
                        region_mask[y1:y2+1, x1:x2+1] = True

                        avg_spec_a = average_spectrum_in_region(
                            st.session_state.change_aligned_a, region_mask
                        )
                        avg_spec_b = average_spectrum_in_region(
                            st.session_state.change_aligned_b, region_mask
                        )

                        wavelengths = st.session_state.change_wavelengths_a
                        if wavelengths is None:
                            wavelengths = np.arange(len(avg_spec_a))

                        diff_stats = compute_spectral_difference(
                            avg_spec_a, avg_spec_b, wavelengths
                        )

                        st.session_state.change_selected_region = {
                            'x1': x1, 'x2': x2,
                            'y1': y1, 'y2': y2,
                            'spectrum_a': avg_spec_a,
                            'spectrum_b': avg_spec_b,
                            'wavelengths': wavelengths,
                            'diff_stats': diff_stats,
                            'n_pixels': int(np.sum(region_mask)),
                        }

                    except Exception as e:
                        st.error(f"❌ 计算区域光谱失败: {str(e)}")

            pixel_data = st.session_state.change_selected_pixel
            region_data = st.session_state.change_selected_region
            display_data = pixel_data if analysis_mode == "单像素光谱对比" else region_data

            if display_data is not None:
                try:
                    st.markdown("#### 📊 光谱曲线对比")

                    spec_a = display_data['spectrum_a']
                    spec_b = display_data['spectrum_b']
                    wl = display_data['wavelengths']
                    diff_stats = display_data['diff_stats']

                    fig_spec = go.Figure()

                    fig_spec.add_trace(go.Scatter(
                        x=wl,
                        y=spec_a,
                        mode='lines',
                        name='时相A',
                        line=dict(color='blue', width=2),
                        hovertemplate='波长: %{x:.1f} nm<br>反射率: %{y:.4f}<extra></extra>'
                    ))

                    fig_spec.add_trace(go.Scatter(
                        x=wl,
                        y=spec_b,
                        mode='lines',
                        name='时相B',
                        line=dict(color='red', width=2),
                        hovertemplate='波长: %{x:.1f} nm<br>反射率: %{y:.4f}<extra></extra>'
                    ))

                    max_band_idx = diff_stats['max_diff_band_index']
                    max_wl = wl[max_band_idx] if len(wl) > max_band_idx else max_band_idx
                    max_diff_val = diff_stats['max_diff_value']

                    fig_spec.add_vline(
                        x=max_wl,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"最大差异波段",
                        annotation_position="top right"
                    )

                    fig_spec.update_layout(
                        title='时相A与时相B光谱曲线对比',
                        xaxis_title='波长 (nm)' if st.session_state.change_wavelengths_a is not None else '波段索引',
                        yaxis_title='反射率',
                        hovermode='x unified',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )

                    st.plotly_chart(fig_spec, use_container_width=True)

                    st.markdown("#### 📉 光谱差异分析")

                    col_sd1, col_sd2, col_sd3, col_sd4 = st.columns(4)
                    col_sd1.metric("最大差异值", f"{diff_stats['max_diff_value']:.4f}")
                    col_sd2.metric("平均差异", f"{diff_stats['mean_diff']:.4f}")
                    col_sd3.metric("光谱角距离 (SAD)", f"{diff_stats['sad']:.4f} rad")
                    col_sd4.metric("欧氏距离", f"{diff_stats['euclidean_distance']:.4f}")

                    if 'max_diff_wavelength' in diff_stats:
                        st.info(f"🌈 光谱差异最大的波段: {diff_stats['max_diff_wavelength']:.1f} nm (波段索引: {diff_stats['max_diff_band_index']})")

                    if analysis_mode == "区域平均光谱对比" and 'n_pixels' in display_data:
                        st.info(f"📐 区域包含像素数: {display_data['n_pixels']:,}")

                    fig_diff = go.Figure()
                    fig_diff.add_trace(go.Scatter(
                        x=wl,
                        y=diff_stats['absolute_difference'],
                        mode='lines',
                        name='绝对差异',
                        fill='tozeroy',
                        line=dict(color='orange', width=2),
                        hovertemplate='波长: %{x:.1f} nm<br>绝对差异: %{y:.4f}<extra></extra>'
                    ))

                    fig_diff.add_vline(
                        x=max_wl,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"最大差异: {max_diff_val:.4f}",
                    )

                    fig_diff.update_layout(
                        title='光谱绝对差异分布',
                        xaxis_title='波长 (nm)' if st.session_state.change_wavelengths_a is not None else '波段索引',
                        yaxis_title='绝对差异',
                        height=400,
                    )

                    st.plotly_chart(fig_diff, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 光谱曲线展示失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        else:
            st.info("💡 导入3期及以上影像（按时间顺序），进行多时相光谱曲线和NDVI时序分析")

            n_phases = st.number_input("时相数量", 3, 20, 3, 1, key='temporal_n_phases')

            phase_uploaded = []
            for i in range(n_phases):
                st.markdown(f"##### 🕒 时相 {i+1}")
                col_h, col_d = st.columns(2)
                with col_h:
                    hdr_f = st.file_uploader(f"上传头文件 (时相{i+1})", type=['hdr'], key=f'temp_hdr_{i}')
                with col_d:
                    dat_f = st.file_uploader(f"上传数据文件 (时相{i+1})", type=['dat', 'raw', 'img'], key=f'temp_dat_{i}')
                phase_uploaded.append((hdr_f, dat_f))

            if st.button("📂 加载多期时序影像", key='load_temporal_btn'):
                temp_images = []
                temp_headers = []
                temp_wavelengths = []
                all_ok = True

                for i, (hdr_f, dat_f) in enumerate(phase_uploaded):
                    if hdr_f is None:
                        st.warning(f"⚠️ 请上传时相 {i+1} 的头文件")
                        all_ok = False
                        continue
                    try:
                        hdr_path = save_uploaded_file(hdr_f)
                        dat_path = None
                        if dat_f is not None:
                            dat_path = save_uploaded_file(dat_f)
                        data, header = load_envi_data(hdr_path, dat_path, mmap=True)
                        temp_images.append(data)
                        temp_headers.append(header)
                        wl = np.array(header.wavelengths) if header.wavelengths else None
                        temp_wavelengths.append(wl)
                        info = get_image_info(data, header)
                        st.success(f"✅ 时相 {i+1} 加载成功: {info['dimensions']}, {info['num_bands']}波段")
                    except Exception as e:
                        st.error(f"❌ 时相 {i+1} 加载失败: {str(e)}")
                        all_ok = False

                if all_ok and len(temp_images) >= 3:
                    st.session_state.temporal_images = temp_images
                    st.session_state.temporal_headers = temp_headers
                    st.session_state.temporal_wavelengths = temp_wavelengths

                    shapes = [img.shape for img in temp_images]
                    all_same = all(s == shapes[0] for s in shapes)

                    if all_same:
                        st.session_state.temporal_aligned = True
                        st.success(f"✅ 所有 {len(temp_images)} 期影像尺寸一致，可直接分析")
                    else:
                        st.warning(f"⚠️ 各期影像尺寸不一致，尝试自动对齐（裁剪到最小尺寸）")
                        min_H = min(s[0] for s in shapes)
                        min_W = min(s[1] for s in shapes)
                        min_B = min(s[2] for s in shapes)

                        aligned_imgs = []
                        for img in temp_images:
                            aligned_imgs.append(img[:min_H, :min_W, :min_B])
                        st.session_state.temporal_images = aligned_imgs
                        st.session_state.temporal_aligned = True
                        st.success(f"✅ 对齐完成，统一尺寸: {min_H} × {min_W} × {min_B}")

            if st.session_state.temporal_aligned and len(st.session_state.temporal_images) >= 3:
                temp_images = st.session_state.temporal_images
                temp_wavelengths = st.session_state.temporal_wavelengths
                n_p = len(temp_images)
                H, W = temp_images[0].shape[:2]

                st.markdown("#### 📊 多时相光谱分析设置")
                temp_analysis_mode = st.radio(
                    "分析对象",
                    ["单像素时序光谱", "区域平均时序光谱"],
                    horizontal=True,
                    key='temp_analysis_mode'
                )

                if temp_analysis_mode == "单像素时序光谱":
                    col_tx, col_ty = st.columns(2)
                    with col_tx:
                        t_x = st.number_input("像素 X 坐标", 0, W-1, W//2, 1, key='t_px_x')
                    with col_ty:
                        t_y = st.number_input("像素 Y 坐标", 0, H-1, H//2, 1, key='t_px_y')
                    analyze_btn = st.button("🔍 分析该像素时序光谱", key='analyze_temp_pixel_btn')
                else:
                    col_tr1, col_tr2, col_tr3, col_tr4 = st.columns(4)
                    with col_tr1:
                        tr_x1 = st.number_input("起始 X", 0, W-1, W//4, 1, key='tr_x1')
                    with col_tr2:
                        tr_x2 = st.number_input("结束 X", 0, W-1, W//2, 1, key='tr_x2')
                    with col_tr3:
                        tr_y1 = st.number_input("起始 Y", 0, H-1, H//4, 1, key='tr_y1')
                    with col_tr4:
                        tr_y2 = st.number_input("结束 Y", 0, H-1, H//2, 1, key='tr_y2')
                    analyze_btn = st.button("🔍 分析区域平均时序光谱", key='analyze_temp_region_btn')

                if analyze_btn:
                    try:
                        with st.spinner("正在计算时序光谱和NDVI..."):
                            spectra_list = []
                            region_info = None

                            if temp_analysis_mode == "单像素时序光谱":
                                for img in temp_images:
                                    spectra_list.append(img[t_y, t_x, :].astype(np.float64))
                            else:
                                x1, x2 = min(tr_x1, tr_x2), max(tr_x1, tr_x2)
                                y1, y2 = min(tr_y1, tr_y2), max(tr_y1, tr_y2)
                                region_mask = np.zeros((H, W), dtype=bool)
                                region_mask[y1:y2+1, x1:x2+1] = True
                                n_pix = int(np.sum(region_mask))
                                region_info = {
                                    'x1': x1, 'x2': x2,
                                    'y1': y1, 'y2': y2,
                                    'n_pixels': n_pix
                                }
                                for img in temp_images:
                                    spectra_list.append(average_spectrum_in_region(img, region_mask))

                            wl = temp_wavelengths[0]
                            if wl is None:
                                wl = np.arange(len(spectra_list[0]))

                            st.session_state.temporal_analysis = {
                                'mode': temp_analysis_mode,
                                'pixel': {'x': t_x, 'y': t_y} if temp_analysis_mode == "单像素时序光谱" else None,
                                'region': region_info,
                                'spectra': spectra_list,
                                'wavelengths': wl,
                                'n_phases': n_p,
                            }

                            if temp_analysis_mode == "单像素时序光谱":
                                ndvi_series = compute_ndvi_pixel_timeseries(
                                    temp_images, t_x, t_y, temp_wavelengths
                                )
                            else:
                                from src.visualization import compute_ndvi as _compute_ndvi
                                ndvi_series = []
                                for i, img in enumerate(temp_images):
                                    ndvi_map = _compute_ndvi(img, temp_wavelengths[i] if i < len(temp_wavelengths) else None)
                                    if temp_analysis_mode == "区域平均时序光谱":
                                        mask = np.zeros((H, W), dtype=bool)
                                        mask[y1:y2+1, x1:x2+1] = True
                                        ndvi_series.append(float(np.mean(ndvi_map[mask])))
                                    else:
                                        ndvi_series.append(float(np.mean(ndvi_map)))

                            st.session_state.temporal_analysis['ndvi_series'] = ndvi_series

                            ndvi_diffs = []
                            for i in range(len(ndvi_series) - 1):
                                ndvi_diffs.append(ndvi_series[i+1] - ndvi_series[i])
                            st.session_state.temporal_analysis['ndvi_diffs'] = ndvi_diffs

                            st.success("✅ 时序分析完成")
                    except Exception as e:
                        st.error(f"❌ 时序分析失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

                if 'temporal_analysis' in st.session_state and st.session_state.temporal_analysis is not None:
                    ta = st.session_state.temporal_analysis

                    st.markdown("#### 🌈 多时相光谱曲线变化轨迹")

                    colors = px.colors.qualitative.Plotly
                    if ta['n_phases'] > len(colors):
                        colors = colors * ((ta['n_phases'] // len(colors)) + 1)

                    fig_ts_spec = go.Figure()

                    for i in range(ta['n_phases']):
                        fig_ts_spec.add_trace(go.Scatter(
                            x=ta['wavelengths'],
                            y=ta['spectra'][i],
                            mode='lines',
                            name=f'时相 {i+1}',
                            line=dict(color=colors[i], width=2),
                            hovertemplate=f'时相{i+1}<br>波长: %{{x:.1f}} nm<br>反射率: %{{y:.4f}}<extra></extra>'
                        ))

                    x_label = '波长 (nm)' if st.session_state.change_wavelengths_a is not None or (ta['wavelengths'] is not None and len(ta['wavelengths']) > 0 and not np.array_equal(ta['wavelengths'], np.arange(len(ta['wavelengths'])))) else '波段索引'

                    fig_ts_spec.update_layout(
                        title=f'多时相光谱曲线 (共{ta["n_phases"]}期)',
                        xaxis_title=x_label,
                        yaxis_title='反射率',
                        hovermode='x unified',
                        height=550,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )

                    st.plotly_chart(fig_ts_spec, use_container_width=True)

                    if ta['mode'] == "单像素时序光谱" and ta['pixel']:
                        st.info(f"📍 分析像素: ({ta['pixel']['x']}, {ta['pixel']['y']})")
                    elif ta['mode'] == "区域平均时序光谱" and ta['region']:
                        st.info(f"📍 分析区域: X[{ta['region']['x1']}-{ta['region']['x2']}], Y[{ta['region']['y1']}-{ta['region']['y2']}], 共{ta['region']['n_pixels']:,}像素")

                    st.markdown("#### 🌿 NDVI时间变化趋势")

                    fig_ndvi = go.Figure()

                    phase_labels = [f'时相 {i+1}' for i in range(ta['n_phases'])]

                    fig_ndvi.add_trace(go.Scatter(
                        x=phase_labels,
                        y=ta['ndvi_series'],
                        mode='lines+markers',
                        name='NDVI值',
                        line=dict(color='green', width=3),
                        marker=dict(size=10, symbol='circle'),
                        hovertemplate='%{x}<br>NDVI: %{y:.4f}<extra></extra>'
                    ))

                    fig_ndvi.update_layout(
                        title='NDVI时间变化趋势折线图',
                        xaxis_title='时相',
                        yaxis_title='NDVI值',
                        height=450,
                        hovermode='x unified',
                    )

                    st.plotly_chart(fig_ndvi, use_container_width=True)

                    if len(ta['ndvi_diffs']) > 0:
                        st.markdown("#### 📊 相邻时相NDVI差值")

                        diff_labels = [f'时相{i+1}→时相{i+2}' for i in range(len(ta['ndvi_diffs']))]
                        bar_colors = ['red' if d < 0 else 'green' for d in ta['ndvi_diffs']]

                        fig_diff_bar = go.Figure()
                        fig_diff_bar.add_trace(go.Bar(
                            x=diff_labels,
                            y=ta['ndvi_diffs'],
                            marker_color=bar_colors,
                            text=[f'{d:+.4f}' for d in ta['ndvi_diffs']],
                            textposition='auto',
                            hovertemplate='%{x}<br>NDVI变化: %{y:+.4f}<extra></extra>'
                        ))

                        fig_diff_bar.add_hline(y=0, line_dash="dash", line_color="gray")

                        fig_diff_bar.update_layout(
                            title='相邻时相NDVI差值（绿=植被增加，红=植被减少）',
                            xaxis_title='时相转换',
                            yaxis_title='NDVI差值',
                            height=400,
                            showlegend=False,
                        )

                        st.plotly_chart(fig_diff_bar, use_container_width=True)

                        col_ndvi1, col_ndvi2, col_ndvi3 = st.columns(3)
                        col_ndvi1.metric("起始时相NDVI", f"{ta['ndvi_series'][0]:.4f}")
                        col_ndvi2.metric("结束时相NDVI", f"{ta['ndvi_series'][-1]:.4f}",
                                        delta=f"{ta['ndvi_series'][-1]-ta['ndvi_series'][0]:+.4f}")
                        col_ndvi3.metric("最大单期变化",
                                        f"{max(ta['ndvi_diffs'], key=abs):+.4f}",
                                        delta="绝对值最大的相邻时相变化")


        st.markdown("---")
        st.subheader("💾 结果下载")

        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

        with col_dl1:
            if st.session_state.change_binary_vis is not None:
                bin_png = (st.session_state.change_binary_vis * 255).astype(np.uint8)
                img = Image.fromarray(bin_png)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                st.download_button(
                    "📥 下载变化二值图 (PNG)",
                    buf.getvalue(),
                    "change_detection_binary.png",
                    "image/png"
                )

        with col_dl2:
            if st.session_state.change_intensity is not None:
                buf = io.BytesIO()
                np.save(buf, st.session_state.change_intensity)
                st.download_button(
                    "📥 下载变化强度数据 (NPY)",
                    buf.getvalue(),
                    "change_intensity.npy",
                    "application/octet-stream"
                )

        with col_dl3:
            if st.session_state.change_stats is not None:
                stats_for_json = {}
                for k, v in st.session_state.change_stats.items():
                    if isinstance(v, np.ndarray):
                        stats_for_json[k] = v.tolist()
                    elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                        stats_for_json[k] = float(v)
                    else:
                        stats_for_json[k] = v
                stats_json = json.dumps(stats_for_json, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 下载统计信息 (JSON)",
                    stats_json,
                    "change_stats.json",
                    "application/json"
                )

        with col_dl4:
            if st.session_state.change_geojson is not None:
                geojson_str = json.dumps(st.session_state.change_geojson, ensure_ascii=False, indent=2)
                st.download_button(
                    "🗺️ 下载变化区域 (GeoJSON)",
                    geojson_str,
                    "change_detection_results.geojson",
                    "application/geo+json"
                )

        if st.session_state.transition_matrix is not None:
            col_dl5, col_dl6 = st.columns(2)

            with col_dl5:
                tm = st.session_state.transition_matrix
                classes = st.session_state.transition_classes
                class_names = st.session_state.samples.class_names
                class_label_list = [class_names.get(c, f'Class {c}') for c in classes]
                tm_df = pd.DataFrame(
                    tm,
                    index=class_label_list,
                    columns=class_label_list
                )
                csv_buf = io.StringIO()
                tm_df.to_csv(csv_buf)
                st.download_button(
                    "📥 下载转移矩阵 (CSV)",
                    csv_buf.getvalue(),
                    "transition_matrix.csv",
                    "text/csv"
                )

else:
    st.info("👆 请先导入两期高光谱影像开始变化检测分析")
