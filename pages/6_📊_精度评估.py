import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from state import init_session_state, create_progress_callback, render_sidebar_info
from src.evaluation import compute_metrics, compute_confusion_matrix, evaluate_classifier, format_metrics_for_display


st.set_page_config(page_title="精度评估", page_icon="📊", layout="wide")
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
