import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from state import init_session_state, create_progress_callback, render_sidebar_info
from src.classification import create_classifier, classify_image
from src.utils import reshape_for_classifier


init_session_state()
render_sidebar_info()

st.header("🤖 模型训练")

if st.session_state.train_samples is None or st.session_state.test_samples is None:
    st.warning("⚠️ 请先完成样本管理和数据集划分")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("选择分类器")
    classifier_type = st.selectbox("分类算法", ["SVM (RBF核)", "随机森林", "1D-CNN", "3D-CNN", "半监督学习（标签传播）"])

with col2:
    st.subheader("算法参数")
    use_unlabeled = False

    if classifier_type == "SVM (RBF核)":
        st.markdown("##### SVM参数")
        enable_grid_search = st.checkbox("启用网格搜索优化", True)
        C_values = st.multiselect("C值候选", [0.01, 0.1, 1, 10, 100], default=[0.1, 1, 10])
        gamma_values = st.multiselect("gamma候选", ['scale', 0.001, 0.01, 0.1, 1], default=['scale', 0.01, 0.1])
        classifier_params = {'C': C_values, 'gamma': gamma_values, 'grid_search': enable_grid_search, 'cv': 3}
        classifier_key = 'svm'

    elif classifier_type == "随机森林":
        st.markdown("##### 随机森林参数")
        n_estimators = st.slider("决策树数量", 10, 500, 100, 10)
        max_depth = st.slider("最大深度", 3, 30, 10)
        max_features = st.selectbox("特征采样比例", ['sqrt', 'log2', 0.3, 0.5, 0.7])
        max_features_param = max_features if isinstance(max_features, str) else float(max_features)
        classifier_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features_param}
        classifier_key = 'random_forest'

    elif classifier_type == "1D-CNN":
        st.markdown("##### 1D-CNN参数")
        n_epochs = st.slider("训练轮数", 10, 200, 50, 10)
        batch_size = st.selectbox("批大小", [32, 64, 128, 256], 2)
        learning_rate = st.selectbox("学习率", [0.0001, 0.001, 0.01, 0.1], 1)
        classifier_params = {'n_epochs': n_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}
        classifier_key = '1d_cnn'

    elif classifier_type == "3D-CNN":
        st.markdown("##### 3D-CNN参数")
        window_size = st.selectbox("空间窗口大小", [3, 5, 7, 9, 11], 2)
        n_epochs = st.slider("训练轮数", 10, 100, 30, 5)
        batch_size = st.selectbox("批大小", [32, 64, 128, 256], 1)
        learning_rate = st.selectbox("学习率", [0.0001, 0.001, 0.01], 1)
        classifier_params = {'window_size': window_size, 'n_epochs': n_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}
        classifier_key = '3d_cnn'

    else:
        st.markdown("##### 半监督学习参数")
        gamma = st.slider("RBF核gamma", 1, 50, 20)
        max_iter = st.slider("最大迭代次数", 100, 5000, 1000, 100)
        n_neighbors = st.slider("邻居数", 3, 15, 7)
        use_unlabeled = st.checkbox("使用无标注样本", True)
        classifier_params = {'gamma': gamma, 'max_iter': max_iter, 'n_neighbors': n_neighbors}
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
                train_info = classifier.fit(train_X, train_y, data_image=data_for_3d, progress_callback=callback)
            elif classifier_key == 'semi_supervised' and use_unlabeled:
                data_for_semi = st.session_state.features if st.session_state.features is not None else st.session_state.preprocessed_data
                data_flat = reshape_for_classifier(data_for_semi)
                n_unlabeled = min(10000, len(data_flat) // 10)
                indices = np.random.choice(len(data_flat), n_unlabeled, replace=False)
                X_unlabeled = data_flat[indices]
                train_info = classifier.fit(train_X, train_y, X_unlabeled=X_unlabeled, progress_callback=callback)
            else:
                train_info = classifier.fit(train_X, train_y, progress_callback=callback)

            st.session_state.classifier = classifier
            st.session_state.train_info = train_info
            progress_bar.progress(1.0)
            status_text.text("✅ 模型训练完成！")
            st.success("✅ 模型训练完成！")

            if 'best_params' in train_info:
                st.info(f"🎯 最优参数: {train_info['best_params']}")

            if 'train_losses' in train_info:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(train_info['train_losses']) + 1)), y=train_info['train_losses'], mode='lines', name='训练损失'))
                fig.add_trace(go.Scatter(x=list(range(1, len(train_info['train_accuracies']) + 1)), y=train_info['train_accuracies'], mode='lines', name='训练精度', yaxis='y2'))
                fig.update_layout(title='训练曲线', xaxis_title='Epoch', yaxis_title='Loss', yaxis2=dict(title='Accuracy', overlaying='y', side='right'), height=400)
                st.plotly_chart(fig, use_container_width=True)

            if 'feature_importances' in train_info:
                importances = train_info['feature_importances']
                feature_names = st.session_state.feature_info.get('feature_names', [f'Feature {i}' for i in range(len(importances))]) if st.session_state.feature_info else [f'Feature {i}' for i in range(len(importances))]
                top_idx = np.argsort(importances)[-20:][::-1]
                fig = px.bar(x=importances[top_idx], y=[feature_names[i] for i in top_idx], orientation='h', title='Top 20 特征重要性')
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
                from src.visualization import classification_to_rgb
                predictions, _ = classify_image(st.session_state.classifier, st.session_state.features, progress_callback=callback)
                st.session_state.classification_result = predictions
                progress_bar.progress(1.0)
                status_text.text("✅ 全图分类完成！")
                st.success("✅ 全图分类完成！")
                class_rgb, legend = classification_to_rgb(predictions, class_names=st.session_state.samples.class_names)
                st.session_state.classification_rgb = class_rgb
                st.session_state.classification_legend = legend
        except Exception as e:
            st.error(f"❌ 分类失败: {str(e)}")
            progress_bar.progress(0)
            status_text.text("")
