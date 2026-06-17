import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import json
from PIL import Image

from state import init_session_state, render_sidebar_info, save_classification_result
from src.visualization import (
    get_true_color, get_false_color, compute_ndvi, colormap_ndvi,
    get_rgb_composite, classification_to_rgb,
    overlay_classification, extract_single_class,
    generate_classification_legend, create_comparison_diff_map
)
from src.utils import normalize_image


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

st.markdown("---")
st.header("🔍 分类结果对比分析")
st.info("💡 选择两个不同的分类结果进行并排对比，查看差异区域")

history = st.session_state.classification_results_history

col_save1, col_save2 = st.columns([1, 1])
with col_save1:
    if st.button("💾 保存当前结果到对比库"):
        default_name = f"结果_{pd.Timestamp.now().strftime('%m%d_%H%M%S')}"
        save_classification_result(
            default_name,
            st.session_state.classification_result,
            st.session_state.classification_rgb,
            st.session_state.classification_legend,
            st.session_state.metrics
        )
        st.success(f"✅ 已保存: {default_name}")

with col_save2:
    if len(history) > 0:
        delete_options = list(history.keys())
        to_delete = st.multiselect("选择要删除的结果", delete_options)
        if st.button("🗑️ 删除选中结果"):
            for name in to_delete:
                del history[name]
            st.success(f"✅ 已删除 {len(to_delete)} 个结果")

if len(history) < 1:
    st.warning("⚠️ 对比库中没有保存的结果。请先执行分类并保存结果。")
    st.info("💡 可以在模型训练页使用不同参数多次分类，保存结果后再进行对比")

col_compare1, col_compare2 = st.columns([1, 1])

with col_compare1:
    options_list = list(history.keys())
    if st.session_state.classification_result is not None:
        options_list = ["[当前结果]"] + options_list

    if len(options_list) >= 2:
        result1_name = st.selectbox("选择结果A（左侧）", options_list, index=0, key="result1_select")
    else:
        result1_name = None

with col_compare2:
    if len(options_list) >= 2:
        default_idx2 = 1 if len(options_list) > 1 else 0
        result2_name = st.selectbox("选择结果B（右侧）", options_list, index=default_idx2, key="result2_select")
    else:
        result2_name = None

if result1_name and result2_name and result1_name != result2_name:
    try:
        if result1_name == "[当前结果]":
            r1_data = st.session_state.classification_result
            r1_rgb = st.session_state.classification_rgb
        else:
            r1_data = history[result1_name]['result']
            r1_rgb = history[result1_name]['rgb']

        if result2_name == "[当前结果]":
            r2_data = st.session_state.classification_result
            r2_rgb = st.session_state.classification_rgb
        else:
            r2_data = history[result2_name]['result']
            r2_rgb = history[result2_name]['rgb']

        if r1_data.shape != r2_data.shape:
            st.error("❌ 两个结果的空间尺寸不匹配，无法对比")
        else:
            diff_bg_type = st.selectbox(
                "差异图背景类型",
                ["真彩色背景", "纯灰背景", "原始影像灰度"],
                key="diff_bg_type"
            )
            diff_alpha = st.slider("差异区域颜色强度", 0.1, 1.0, 0.5, 0.05, key="diff_alpha")

            if diff_bg_type == "真彩色背景":
                diff_bg = get_true_color(st.session_state.data, st.session_state.wavelengths)
            elif diff_bg_type == "原始影像灰度":
                max_band_idx = st.session_state.data.shape[2] - 1
                band_idx = min(30, max_band_idx)
                single_band = normalize_image(st.session_state.data[:, :, band_idx])
                diff_bg = np.stack([single_band] * 3, axis=-1)
            else:
                diff_bg = None

            diff_map, diff_stats = create_comparison_diff_map(
                r1_data, r2_data, background=diff_bg, alpha=diff_alpha
            )

            st.markdown("---")
            col_stat_c1, col_stat_c2, col_stat_c3, col_stat_c4 = st.columns(4)
            col_stat_c1.metric("总像素数", f"{diff_stats['total_pixels']:,}")
            col_stat_c2.metric("✅ 分类一致像素", f"{diff_stats['agree_pixels']:,}")
            col_stat_c3.metric("❌ 分类差异像素", f"{diff_stats['disagree_pixels']:,}")
            col_stat_c4.metric("📊 一致率", f"{diff_stats['agreement_ratio']*100:.2f}%")

            st.markdown("---")
            st.subheader("🖼️ 分类结果并排对比")

            display_width = 8
            fig_side, axes_side = plt.subplots(1, 3, figsize=(display_width * 3, display_width))
            axes_side[0].imshow(r1_rgb)
            axes_side[0].set_title(f'结果A: {result1_name}', fontsize=16)
            axes_side[0].axis('off')

            axes_side[1].imshow(r2_rgb)
            axes_side[1].set_title(f'结果B: {result2_name}', fontsize=16)
            axes_side[1].axis('off')

            axes_side[2].imshow(diff_map)
            axes_side[2].set_title('差异图（灰显=一致，彩色=差异）', fontsize=16)
            axes_side[2].axis('off')

            plt.tight_layout()
            st.pyplot(fig_side)

            st.markdown("---")
            st.subheader("📊 差异详细分析")

            agree_mask = r1_data == r2_data
            disagree_mask = ~agree_mask

            classes_r1 = np.unique(r1_data)
            classes_r2 = np.unique(r2_data)
            all_classes = sorted(np.unique(np.concatenate([classes_r1, classes_r2])).tolist())

            class_compare_data = []
            for cls in all_classes:
                name = class_names.get(cls, f"Class {cls}")
                count1 = int(np.sum(r1_data == cls))
                count2 = int(np.sum(r2_data == cls))
                diff_count = count1 - count2
                diff_pct = (diff_count / max(count1, 1)) * 100
                class_compare_data.append({
                    '类别': name,
                    f'{result1_name}像素数': count1,
                    f'{result2_name}像素数': count2,
                    '差异数': diff_count,
                    '差异比例': f"{diff_pct:+.2f}%"
                })
            class_compare_df = pd.DataFrame(class_compare_data)
            st.dataframe(class_compare_df, use_container_width=True, height=300)

            sorted_pairs = []
            if diff_stats['disagree_pixels'] > 0:
                st.markdown("---")
                st.subheader("🔎 差异模式分析")

                disagree_r1 = r1_data[disagree_mask]
                disagree_r2 = r2_data[disagree_mask]

                diff_pairs = {}
                for i in range(len(disagree_r1)):
                    c1, c2 = int(disagree_r1[i]), int(disagree_r2[i])
                    key = (c1, c2)
                    if key not in diff_pairs:
                        diff_pairs[key] = 0
                    diff_pairs[key] += 1

                sorted_pairs = sorted(diff_pairs.items(), key=lambda x: x[1], reverse=True)
                top_diff_n = min(15, len(sorted_pairs))

                pair_diff_data = []
                for rank_idx, ((c1, c2), cnt) in enumerate(sorted_pairs[:top_diff_n]):
                    n1 = class_names.get(c1, f"Class {c1}")
                    n2 = class_names.get(c2, f"Class {c2}")
                    total_diff = diff_stats['disagree_pixels']
                    ratio = cnt / total_diff * 100 if total_diff > 0 else 0
                    pair_diff_data.append({
                        '排名': rank_idx + 1,
                        f'{result1_name}分类为': n1,
                        f'{result2_name}分类为': n2,
                        '差异像素数': cnt,
                        '占总差异比例': f"{ratio:.2f}%"
                    })
                pair_diff_df = pd.DataFrame(pair_diff_data)
                st.dataframe(pair_diff_df, use_container_width=True, height=300)

                st.markdown("---")
                st.markdown("**📊 主要差异分布**")

                pair_names = [
                    f"{class_names.get(p[0], f'C{p[0]}')}↔{class_names.get(p[1], f'C{p[1]}')}"
                    for (p, _) in sorted_pairs[:top_diff_n]
                ]
                pair_counts = [c for (_, c) in sorted_pairs[:top_diff_n]]

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=pair_names,
                    y=pair_counts,
                    text=[f"{c}个" for c in pair_counts],
                    textposition='auto',
                    marker_color=px.colors.sequential.RdBu_r[:len(pair_names)]
                ))
                fig_bar.update_layout(
                    title=f'Top {top_diff_n} 差异类别对分布',
                    xaxis_title='差异类别对',
                    yaxis_title='差异像素数',
                    height=500,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.subheader("💾 下载对比结果")
            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                diff_png = (diff_map * 255).astype(np.uint8)
                diff_img = Image.fromarray(diff_png)
                diff_buf = io.BytesIO()
                diff_img.save(diff_buf, format='PNG')
                st.download_button(
                    "📥 下载差异图 (PNG)",
                    diff_buf.getvalue(),
                    f"diff_map_{result1_name[:10]}_vs_{result2_name[:10]}.png",
                    "image/png"
                )

            with col_dl2:
                diff_npy_buf = io.BytesIO()
                np.save(diff_npy_buf, disagree_mask.astype(np.uint8))
                st.download_button(
                    "📥 下载差异掩码 (NPY)",
                    diff_npy_buf.getvalue(),
                    f"diff_mask_{result1_name[:10]}_vs_{result2_name[:10]}.npy",
                    "application/octet-stream"
                )

            with col_dl3:
                top_pairs_for_report = sorted_pairs[:min(50, len(sorted_pairs))] if sorted_pairs else []
                compare_report = {
                    'result_A': result1_name,
                    'result_B': result2_name,
                    'stats': diff_stats,
                    'per_class_difference': [
                        {
                            'class': int(c),
                            'class_name': class_names.get(c, f'Class {c}'),
                            'count_A': int(np.sum(r1_data == c)),
                            'count_B': int(np.sum(r2_data == c))
                        }
                        for c in all_classes
                    ],
                    'top_difference_pairs': [
                        {
                            'class_A': int(p[0]),
                            'class_B': int(p[1]),
                            'class_A_name': class_names.get(p[0], f'Class {p[0]}'),
                            'class_B_name': class_names.get(p[1], f'Class {p[1]}'),
                            'count': int(c)
                        }
                        for (p, c) in top_pairs_for_report
                    ]
                }
                report_json = json.dumps(compare_report, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 下载对比报告 (JSON)",
                    report_json,
                    f"comparison_report_{result1_name[:10]}_vs_{result2_name[:10]}.json",
                    "application/json"
                )

    except Exception as e:
        st.error(f"❌ 对比分析失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
