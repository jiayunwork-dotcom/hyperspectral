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
    st.info("💡 红色区域表示分类错误的像素，鼠标悬停可查看该像素的真实类别和预测类别")

    if st.session_state.samples is not None and st.session_state.classification_result is not None:
        col_err1, col_err2 = st.columns([1, 1])
        with col_err1:
            error_alpha = st.slider("错误区域透明度", 0.1, 1.0, 0.6, 0.05, key="error_alpha")
            bg_type = st.selectbox("背景影像", ["真彩色", "标准假彩色"], key="error_bg")
        with col_err2:
            show_only_errors = st.checkbox("仅显示错误区域", False, key="show_only_errors")
            show_sample_mode = st.selectbox("显示方式", ["全部标注点", "仅错误点"], key="show_sample_mode")
            error_pixel_count = 0

        try:
            samples = st.session_state.samples
            class_pred = st.session_state.classification_result
            H, W = class_pred.shape
            label_map = np.zeros((H, W), dtype=np.int32)
            pred_at_samples = np.zeros((H, W), dtype=np.int32)
            valid_mask = np.zeros((H, W), dtype=bool)

            if hasattr(samples, 'locations') and samples.locations is not None and samples.locations is not None:
                valid_rows = []
                valid_cols = []
                valid_labels = []
                for i, (r, c) in enumerate(samples.locations):
                    r_int = int(r) if not isinstance(r, int) else r
                    c_int = int(c) if not isinstance(c, int) else c
                    if 0 <= r_int < H and 0 <= c_int < W:
                        valid_rows.append(r_int)
                        valid_cols.append(c_int)
                        valid_labels.append(samples.labels[i])
                        label_map[r_int, c_int] = samples.labels[i]
                        pred_at_samples[r_int, c_int] = class_pred[r_int, c_int]
                        valid_mask[r_int, c_int] = True
            else:
                all_labels = np.concatenate([
                    st.session_state.train_samples.labels,
                    st.session_state.test_samples.labels
                ])
                rng = np.random.RandomState(42)
                n_all = len(all_labels)
                rows = rng.randint(0, H, n_all)
                cols = rng.randint(0, W, n_all)
                valid_rows = []
                valid_cols = []
                valid_labels = []
                for i in range(n_all):
                    r, c = int(rows[i]), int(cols[i])
                    if 0 <= r < H and 0 <= c < W:
                        valid_rows.append(r)
                        valid_cols.append(c)
                        valid_labels.append(all_labels[i])
                        label_map[r, c] = all_labels[i]
                        pred_at_samples[r, c] = class_pred[r, c]
                        valid_mask[r, c] = True

            wavelengths = st.session_state.wavelengths
            if bg_type == "真彩色":
                background = get_true_color(st.session_state.data, wavelengths)
            else:
                from src.visualization import get_false_color
                background = get_false_color(st.session_state.data, wavelengths)

            error_mask = (label_map != pred_at_samples) & valid_mask
            correct_mask = (label_map == pred_at_samples) & valid_mask

            error_pixel_count = int(np.sum(error_mask))
            valid_pixel_count = int(np.sum(valid_mask))
            error_rate = error_pixel_count / valid_pixel_count if valid_pixel_count > 0 else 0

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("有效标注像素", valid_pixel_count)
            col_stat2.metric("分类错误像素", error_pixel_count)
            col_stat3.metric("错误率", f"{error_rate*100:.2f}%")

            class_names = st.session_state.samples.class_names

            error_rows, error_cols = np.where(error_mask)
            correct_rows, correct_cols = np.where(correct_mask)

            error_hover_texts = []
            for i in range(len(error_rows)):
                r, c = error_rows[i], error_cols[i]
                true_cls = int(label_map[r, c])
                pred_cls = int(pred_at_samples[r, c])
                true_name = class_names.get(true_cls, f"Class {true_cls}")
                pred_name = class_names.get(pred_cls, f"Class {pred_cls}")
                error_hover_texts.append(
                    f"位置: ({r}, {c})<br>真实类别: {true_name}<br>预测类别: {pred_name}<br>状态: ❌ 错误"
                )

            correct_hover_texts = []
            for i in range(len(correct_rows)):
                r, c = correct_rows[i], correct_cols[i]
                true_cls = int(label_map[r, c])
                true_name = class_names.get(true_cls, f"Class {true_cls}")
                correct_hover_texts.append(
                    f"位置: ({r}, {c})<br>类别: {true_name}<br>状态: ✅ 正确"
                )

            fig = go.Figure()

            fig.add_trace(go.Image(
                z=(background * 255).astype(np.uint8),
                hoverinfo='none',
                name='背景影像'
            ))

            if show_sample_mode in ["全部标注点"] and len(correct_rows) > 0:
                max_correct_show = min(5000, len(correct_rows))
                sample_idx = np.random.choice(len(correct_rows), max_correct_show, replace=False)
                fig.add_trace(go.Scatter(
                    x=correct_cols[sample_idx],
                    y=correct_rows[sample_idx],
                    mode='markers',
                    marker=dict(size=4, color='limegreen', opacity=0.8, line=dict(width=0.5, color='white')),
                    text=[correct_hover_texts[i] for i in sample_idx],
                    hovertemplate='%{text}<extra></extra>',
                    name='正确分类点'
                ))

            if len(error_rows) > 0:
                fig.add_trace(go.Scatter(
                    x=error_cols,
                    y=error_rows,
                    mode='markers',
                    marker=dict(size=6, color='red', opacity=error_alpha, line=dict(width=1, color='darkred')),
                    text=error_hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    name='错误分类点'
                ))

            fig.update_layout(
                title='分类错误空间分布（鼠标悬停查看详情）',
                xaxis_title='列 (像素)',
                yaxis_title='行 (像素)',
                yaxis=dict(autorange='reversed'),
                height=600,
                hovermode='closest',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            if error_pixel_count > 0:
                st.markdown("---")
                st.markdown("**📋 错误像素详情表**")

                error_data = []
                show_n = min(50, len(error_rows))
                sample_indices = np.random.choice(len(error_rows), show_n, replace=False)
                for idx in sample_indices:
                    r, c = error_rows[idx], error_cols[idx]
                    true_cls = int(label_map[r, c])
                    pred_cls = int(pred_at_samples[r, c])
                    true_name = class_names.get(true_cls, f"Class {true_cls}")
                    pred_name = class_names.get(pred_cls, f"Class {pred_cls}")
                    error_data.append({
                        '行': r,
                        '列': c,
                        '真实类别': true_name,
                        '预测类别': pred_name
                    })
                error_df = pd.DataFrame(error_data)
                st.dataframe(error_df, use_container_width=True, height=250)
                st.caption(f"💡 共 {error_pixel_count} 个错误像素，上方随机展示 {show_n} 个")

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
