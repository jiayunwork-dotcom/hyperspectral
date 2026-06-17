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

        st.info("💡 点击图像或框选区域，查看光谱曲线变化分析")

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


        st.markdown("---")
        st.subheader("💾 结果下载")

        col_dl1, col_dl2, col_dl3 = st.columns(3)

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

        if st.session_state.transition_matrix is not None:
            col_dl4, col_dl5 = st.columns(2)

            with col_dl4:
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
