import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from state import init_session_state, create_progress_callback, render_sidebar_info, compute_class_confusion_pairs
from src.evaluation import compute_metrics, compute_confusion_matrix, evaluate_classifier, format_metrics_for_display
from src.visualization import create_error_spatial_map, get_true_color, normalize_image


init_session_state()
render_sidebar_info()

st.header("📊 精度评估")

if st.session_state.classifier is None or st.session_state.test_samples is None:
    st.warning("⚠️ 请先完成模型训练和数据集划分")
    st.stop()

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
    normalize_cm = st.checkbox("归一化显示", True)
    cm_result = compute_confusion_matrix(
        st.session_state.y_true, st.session_state.y_pred,
        normalize=normalize_cm, class_names=st.session_state.samples.class_names
    )
    cm_df = pd.DataFrame(cm_result['matrix'], index=cm_result['tick_labels'], columns=cm_result['tick_labels'])
    fig = px.imshow(
        cm_df, text_auto='.2f' if normalize_cm else 'd',
        color_continuous_scale='Blues',
        title=f"混淆矩阵 {'(归一化)' if normalize_cm else '(原始计数)'}",
        aspect='auto'
    )
    fig.update_layout(xaxis_title='预测类别', yaxis_title='真实类别', height=600)
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
    fig.update_layout(title='各类别精度指标对比', xaxis_title='类别', yaxis_title='分数', barmode='group', yaxis_range=[0, 1.05], height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔴 错误空间分布分析")
    st.info("💡 红色区域表示分类错误的像素，叠加在原始影像上展示")

    if st.session_state.samples is not None and st.session_state.classification_result is not None:
        col_err1, col_err2 = st.columns([1, 1])
        with col_err1:
            error_alpha = st.slider("错误区域透明度", 0.1, 1.0, 0.6, 0.05, key="error_alpha")
            bg_type = st.selectbox("背景影像", ["真彩色", "标准假彩色"], key="error_bg")
        with col_err2:
            show_only_errors = st.checkbox("仅显示错误区域", False, key="show_only_errors")
            error_pixel_count = 0

        try:
            samples = st.session_state.samples
            class_pred = st.session_state.classification_result
            H, W = class_pred.shape
            label_map = np.zeros((H, W), dtype=np.int32)
            pred_at_samples = np.zeros((H, W), dtype=np.int32)
            valid_mask = np.zeros((H, W), dtype=bool)

            if hasattr(samples, 'locations') and samples.locations is not None:
                for i, (r, c) in enumerate(samples.locations):
                    if 0 <= r < H and 0 <= c < W:
                        label_map[r, c] = samples.labels[i]
                        pred_at_samples[r, c] = class_pred[r, c]
                        valid_mask[r, c] = True
            else:
                all_labels = np.concatenate([
                    st.session_state.train_samples.labels,
                    st.session_state.test_samples.labels
                ])
                from src.classification import train_test_split as split_fn
                rng = np.random.RandomState(42)
                n_all = len(all_labels)
                rows = rng.randint(0, H, n_all)
                cols = rng.randint(0, W, n_all)
                for i in range(n_all):
                    r, c = rows[i], cols[i]
                    if 0 <= r < H and 0 <= c < W:
                        label_map[r, c] = all_labels[i]
                        pred_at_samples[r, c] = class_pred[r, c]
                        valid_mask[r, c] = True

            wavelengths = st.session_state.wavelengths
            if bg_type == "真彩色":
                background = get_true_color(st.session_state.data, wavelengths)
            else:
                from src.visualization import get_false_color
                background = get_false_color(st.session_state.data, wavelengths)

            error_display, error_mask = create_error_spatial_map(
                label_map, pred_at_samples,
                background=background, alpha=error_alpha
            )

            if show_only_errors:
                only_errors = np.ones_like(error_display) * 0.9
                only_errors[error_mask] = error_display[error_mask]
                error_display = only_errors

            error_pixel_count = int(np.sum(error_mask))
            valid_pixel_count = int(np.sum(valid_mask))
            error_rate = error_pixel_count / valid_pixel_count if valid_pixel_count > 0 else 0

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("有效标注像素", valid_pixel_count)
            col_stat2.metric("分类错误像素", error_pixel_count)
            col_stat3.metric("错误率", f"{error_rate*100:.2f}%")

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(error_display)
            ax.set_title('分类错误空间分布（红色为错误区域）')
            ax.axis('off')
            st.pyplot(fig)

            if error_pixel_count > 0:
                st.markdown("---")
                st.markdown("**💡 交互式错误像素信息（点击图片区域查看）**")
                st.caption("提示：从上图可见红色区域集中的位置为模型易分错的区域")

                error_rows, error_cols = np.where(error_mask)
                if len(error_rows) > 0:
                    show_n = min(50, len(error_rows))
                    sample_indices = np.random.choice(len(error_rows), show_n, replace=False)
                    error_data = []
                    for idx in sample_indices:
                        r, c = error_rows[idx], error_cols[idx]
                        true_cls = label_map[r, c]
                        pred_cls = pred_at_samples[r, c]
                        true_name = st.session_state.samples.class_names.get(true_cls, f"Class {true_cls}")
                        pred_name = st.session_state.samples.class_names.get(pred_cls, f"Class {pred_cls}")
                        error_data.append({
                            '行': r,
                            '列': c,
                            '真实类别': true_name,
                            '预测类别': pred_name
                        })
                    error_df = pd.DataFrame(error_data)
                    st.dataframe(error_df, use_container_width=True, height=250)

        except Exception as e:
            st.warning(f"⚠️ 无法生成错误空间分布图: {str(e)}")
            import traceback
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())

    st.markdown("---")
    st.subheader("⚡ 类别间混淆度排名")
    st.info("💡 列出最容易互相混淆的类别对，帮助理解模型的误分类模式")

    if st.session_state.y_true is not None and st.session_state.y_pred is not None:
        try:
            confusion_pairs = compute_class_confusion_pairs(
                st.session_state.y_true,
                st.session_state.y_pred,
                st.session_state.samples.class_names
            )
            st.session_state.class_confusion_pairs = confusion_pairs

            top_n = st.slider("显示Top N混淆对", 5, min(50, len(confusion_pairs)) if len(confusion_pairs) > 0 else 5,
                             min(10, len(confusion_pairs)) if len(confusion_pairs) > 0 else 5,
                             5, key="top_n_confusion")

            if len(confusion_pairs) > 0:
                top_pairs = confusion_pairs[:top_n]

                pair_data = []
                for pair in top_pairs:
                    pair_data.append({
                        '排名': top_pairs.index(pair) + 1,
                        '真实类别': pair['true_name'],
                        '预测为': pair['pred_name'],
                        '混淆数量': pair['count'],
                        '混淆比例': f"{pair['confusion_ratio']*100:.2f}%"
                    })
                pair_df = pd.DataFrame(pair_data)
                st.dataframe(pair_df, use_container_width=True, height=350)

                st.markdown("---")
                st.markdown("**📊 混淆对可视化**")
                pair_names = [f"{p['true_name']}→{p['pred_name']}" for p in top_pairs]
                pair_ratios = [p['confusion_ratio'] * 100 for p in top_pairs]
                pair_counts = [p['count'] for p in top_pairs]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pair_names,
                    y=pair_ratios,
                    text=[f"{c}个 ({r:.1f}%)" for c, r in zip(pair_counts, pair_ratios)],
                    textposition='auto',
                    marker_color=px.colors.sequential.Reds[::-1][:len(top_pairs)]
                ))
                fig.update_layout(
                    title=f'Top {top_n} 类别混淆对（混淆比例）',
                    xaxis_title='混淆对（真实→预测）',
                    yaxis_title='混淆比例 (%)',
                    height=500,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("**🔍 类别间双向混淆矩阵**")
                bidirectional_pairs = []
                for i in range(len(confusion_pairs)):
                    for j in range(i + 1, len(confusion_pairs)):
                        p1, p2 = confusion_pairs[i], confusion_pairs[j]
                        if (p1['true_class'] == p2['pred_class'] and
                            p1['pred_class'] == p2['true_class']):
                            bidirectional_pairs.append({
                                '类别A': p1['true_name'],
                                '类别B': p1['pred_name'],
                                'A→B混淆数': p1['count'],
                                'A→B混淆率': f"{p1['confusion_ratio']*100:.2f}%",
                                'B→A混淆数': p2['count'],
                                'B→A混淆率': f"{p2['confusion_ratio']*100:.2f}%",
                                '总混淆数': p1['count'] + p2['count']
                            })

                if len(bidirectional_pairs) > 0:
                    bidirectional_pairs.sort(key=lambda x: x['总混淆数'], reverse=True)
                    bid_df = pd.DataFrame(bidirectional_pairs[:min(20, len(bidirectional_pairs))])
                    st.dataframe(bid_df, use_container_width=True)
                    st.caption("💡 双向混淆表示两个类别在特征空间中高度相似，建议增加区分性特征或样本")
                else:
                    st.info("✅ 没有发现显著的双向混淆类别对")
            else:
                st.success("✅ 完美！没有发现任何分类混淆")

        except Exception as e:
            st.warning(f"⚠️ 无法计算类别混淆度: {str(e)}")
            import traceback
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())
