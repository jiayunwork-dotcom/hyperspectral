import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import product

from state import init_session_state, create_progress_callback, render_sidebar_info, save_classification_result
from src.classification import (
    create_classifier, classify_image, run_hyperparameter_experiment,
    generate_param_grid_linear, generate_param_grid_log
)
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
    col_classify1, col_classify2 = st.columns([1, 1])
    with col_classify1:
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

    with col_classify2:
        save_name = st.text_input("保存结果名称", value=f"{classifier_type.split(' ')[0]}_{pd.Timestamp.now().strftime('%H%M%S')}")
        if st.button("💾 保存当前结果到对比历史"):
            if st.session_state.classification_result is not None:
                save_classification_result(
                    save_name,
                    st.session_state.classification_result,
                    st.session_state.classification_rgb,
                    st.session_state.classification_legend,
                    st.session_state.metrics
                )
                st.success(f"✅ 已保存: {save_name}")
            else:
                st.warning("⚠️ 请先执行全图分类")

st.markdown("---")
st.subheader("🧪 超参数对比实验")
st.info("💡 配置多组超参数，系统将自动遍历所有组合并对比结果")

exp_classifier = st.selectbox(
    "选择实验分类器",
    ["SVM (RBF核)", "随机森林", "1D-CNN", "3D-CNN"],
    key="exp_classifier"
)
exp_val_ratio = st.slider("验证集比例（从训练集中划分）", 0.1, 0.4, 0.2, 0.05)

if exp_classifier == "SVM (RBF核)":
    st.markdown("##### SVM超参数网格")
    col_svm1, col_svm2, col_svm3 = st.columns(3)
    with col_svm1:
        svm_c_mode = st.radio("C值采样方式", ["线性", "对数"], key="svm_c_mode")
        svm_c_start = st.number_input("C起始值", 0.01, 1000.0, 0.1, key="svm_c_start")
        svm_c_end = st.number_input("C结束值", 0.01, 1000.0, 100.0, key="svm_c_end")
        svm_c_n = st.slider("C采样数", 2, 10, 5, key="svm_c_n")
    with col_svm2:
        svm_g_mode = st.radio("gamma采样方式", ["对数", "固定值"], key="svm_g_mode")
        if svm_g_mode == "对数":
            svm_g_start = st.number_input("gamma起始值", 0.0001, 100.0, 0.001, key="svm_g_start")
            svm_g_end = st.number_input("gamma结束值", 0.0001, 100.0, 1.0, key="svm_g_end")
            svm_g_n = st.slider("gamma采样数", 2, 10, 4, key="svm_g_n")
            gamma_list = generate_param_grid_log(svm_g_start, svm_g_end, svm_g_n)
        else:
            gamma_manual = st.multiselect("gamma值", ['scale', 0.001, 0.01, 0.1, 1, 10], default=['scale', 0.01, 0.1], key="svm_g_manual")
            gamma_list = gamma_manual
    with col_svm3:
        st.markdown("**参数组合预览**")
        if svm_c_mode == "线性":
            c_list = generate_param_grid_linear(svm_c_start, svm_c_end, svm_c_n)
        else:
            c_list = generate_param_grid_log(svm_c_start, svm_c_end, svm_c_n)
        total_combos = len(c_list) * len(gamma_list)
        st.metric("C值数量", len(c_list))
        st.metric("gamma数量", len(gamma_list))
        st.metric("总组合数", total_combos)
        exp_param_grid = {'C': c_list, 'gamma': gamma_list}
        exp_classifier_key = 'svm'

elif exp_classifier == "随机森林":
    st.markdown("##### 随机森林超参数网格")
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        rf_n_start = st.slider("n_estimators起始", 10, 500, 50, 10, key="rf_n_start")
        rf_n_end = st.slider("n_estimators结束", 10, 500, 200, 10, key="rf_n_end")
        rf_n_n = st.slider("n_estimators采样数", 2, 6, 3, key="rf_n_n")
        n_est_list = [int(x) for x in generate_param_grid_linear(rf_n_start, rf_n_end, rf_n_n)]
    with col_rf2:
        rf_d_start = st.slider("max_depth起始", 3, 50, 5, 1, key="rf_d_start")
        rf_d_end = st.slider("max_depth结束", 3, 50, 25, 1, key="rf_d_end")
        rf_d_n = st.slider("max_depth采样数", 2, 6, 3, key="rf_d_n")
        max_depth_list = [int(x) for x in generate_param_grid_linear(rf_d_start, rf_d_end, rf_d_n)]
    with col_rf3:
        rf_feat = st.multiselect("max_features", ['sqrt', 'log2', 0.3, 0.5, 0.7], default=['sqrt', 'log2', 0.5], key="rf_feat")
        total_combos = len(n_est_list) * len(max_depth_list) * len(rf_feat)
        st.metric("总组合数", total_combos)
        exp_param_grid = {'n_estimators': n_est_list, 'max_depth': max_depth_list, 'max_features': rf_feat}
        exp_classifier_key = 'random_forest'

elif exp_classifier == "1D-CNN":
    st.markdown("##### 1D-CNN超参数网格")
    st.warning("⚠️ CNN训练时间较长，建议减少组合数量或使用较少的epoch")
    col_cnn1, col_cnn2, col_cnn3 = st.columns(3)
    with col_cnn1:
        cnn_epochs_list = st.multiselect(
            "n_epochs (训练轮数)",
            [10, 20, 30, 50, 80, 100],
            default=[10, 20, 30],
            key="cnn_epochs"
        )
    with col_cnn2:
        cnn_lr_mode = st.radio("学习率采样方式", ["对数", "固定值"], key="cnn_lr_mode")
        if cnn_lr_mode == "对数":
            cnn_lr_start = st.number_input("学习率起始", 0.0001, 0.1, 0.0005, key="cnn_lr_start", format="%.4f")
            cnn_lr_end = st.number_input("学习率结束", 0.0001, 0.1, 0.01, key="cnn_lr_end", format="%.4f")
            cnn_lr_n = st.slider("学习率采样数", 2, 6, 3, key="cnn_lr_n")
            lr_list = generate_param_grid_log(cnn_lr_start, cnn_lr_end, cnn_lr_n)
        else:
            lr_manual = st.multiselect(
                "学习率值",
                [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                default=[0.0005, 0.001, 0.005],
                key="cnn_lr_manual"
            )
            lr_list = lr_manual
    with col_cnn3:
        cnn_batch_list = st.multiselect(
            "batch_size (批次大小)",
            [64, 128, 256, 512],
            default=[128, 256],
            key="cnn_batch"
        )
        total_combos = len(cnn_epochs_list) * len(lr_list) * len(cnn_batch_list)
        st.metric("总组合数", total_combos)
        exp_param_grid = {
            'n_epochs': cnn_epochs_list,
            'learning_rate': lr_list,
            'batch_size': cnn_batch_list
        }
        exp_classifier_key = '1d_cnn'

elif exp_classifier == "3D-CNN":
    st.markdown("##### 3D-CNN超参数网格")
    st.warning("⚠️ 3D-CNN训练时间较长，建议减少组合数量或使用较少的epoch")
    col_3d1, col_3d2, col_3d3 = st.columns(3)
    with col_3d1:
        cnn3d_epochs_list = st.multiselect(
            "n_epochs (训练轮数)",
            [5, 10, 15, 20, 30],
            default=[5, 10, 15],
            key="3dcnn_epochs"
        )
        cnn3d_ws_list = st.multiselect(
            "window_size (窗口大小)",
            [3, 5, 7, 9],
            default=[5, 7],
            key="3dcnn_ws"
        )
    with col_3d2:
        cnn3d_lr_mode = st.radio("学习率采样方式", ["对数", "固定值"], key="3dcnn_lr_mode")
        if cnn3d_lr_mode == "对数":
            cnn3d_lr_start = st.number_input("学习率起始", 0.0001, 0.01, 0.0005, key="3dcnn_lr_start", format="%.4f")
            cnn3d_lr_end = st.number_input("学习率结束", 0.0001, 0.01, 0.005, key="3dcnn_lr_end", format="%.4f")
            cnn3d_lr_n = st.slider("学习率采样数", 2, 5, 3, key="3dcnn_lr_n")
            lr3d_list = generate_param_grid_log(cnn3d_lr_start, cnn3d_lr_end, cnn3d_lr_n)
        else:
            lr3d_manual = st.multiselect(
                "学习率值",
                [0.0001, 0.0005, 0.001, 0.003, 0.005],
                default=[0.0005, 0.001],
                key="3dcnn_lr_manual"
            )
            lr3d_list = lr3d_manual
    with col_3d3:
        cnn3d_batch_list = st.multiselect(
            "batch_size (批次大小)",
            [32, 64, 128],
            default=[64, 128],
            key="3dcnn_batch"
        )
        total_combos = len(cnn3d_epochs_list) * len(lr3d_list) * len(cnn3d_batch_list) * len(cnn3d_ws_list)
        st.metric("总组合数", total_combos)
        exp_param_grid = {
            'n_epochs': cnn3d_epochs_list,
            'learning_rate': lr3d_list,
            'batch_size': cnn3d_batch_list,
            'window_size': cnn3d_ws_list
        }
        exp_classifier_key = '3d_cnn'

if st.button("⚡ 开始超参数实验", type='primary'):
    if total_combos > 30 and exp_classifier in ["1D-CNN", "3D-CNN"]:
        st.warning(f"⚠️ CNN组合数较多({total_combos})，训练时间可能很长，建议减少组合数")

    progress_bar = st.progress(0)
    status_text = st.empty()
    callback = create_progress_callback(progress_bar, status_text)

    try:
        from src.classification import train_test_split
        X_train_all = st.session_state.train_samples.features
        y_train_all = st.session_state.train_samples.labels
        train_locs_all = st.session_state.train_samples.locations

        n_samples = len(y_train_all)
        indices = np.arange(n_samples)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        n_val = int(n_samples * exp_val_ratio)
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]

        X_tr = X_train_all[tr_idx]
        y_tr = y_train_all[tr_idx]
        X_val = X_train_all[val_idx]
        y_val = y_train_all[val_idx]

        extra_kwargs = {}
        if exp_classifier_key == '3d_cnn':
            extra_kwargs['data_image'] = st.session_state.data
            if train_locs_all is not None:
                train_locs_all = np.array(train_locs_all, dtype=np.int32)
                extra_kwargs['train_locations'] = train_locs_all[tr_idx]
                extra_kwargs['val_locations'] = train_locs_all[val_idx]
            else:
                H, W, _ = st.session_state.data.shape
                half_ws = 7
                tr_locs = np.column_stack([
                    rng.randint(half_ws, H - half_ws, len(y_tr)),
                    rng.randint(half_ws, W - half_ws, len(y_tr))
                ])
                val_locs = np.column_stack([
                    rng.randint(half_ws, H - half_ws, len(y_val)),
                    rng.randint(half_ws, W - half_ws, len(y_val))
                ])
                extra_kwargs['train_locations'] = tr_locs
                extra_kwargs['val_locations'] = val_locs

        with st.spinner("正在运行超参数对比实验..."):
            results = run_hyperparameter_experiment(
                exp_classifier_key,
                X_tr, y_tr, X_val, y_val,
                exp_param_grid,
                progress_callback=callback,
                class_names=st.session_state.samples.class_names,
                **extra_kwargs
            )

        st.session_state.hyperparam_results = results
        st.session_state.hyperparam_best_idx = None

        progress_bar.progress(1.0)
        status_text.text("✅ 超参数实验完成！")
        st.success(f"✅ 实验完成！共测试 {len(results)} 组参数")

        has_errors = any('error' in r for r in results)
        if has_errors:
            error_count = sum(1 for r in results if 'error' in r)
            st.warning(f"⚠️ 有 {error_count} 组参数训练失败，可展开结果表格查看详情")

    except Exception as e:
        st.error(f"❌ 实验失败: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")
        import traceback
        st.code(traceback.format_exc())

if st.session_state.hyperparam_results is not None:
    results = st.session_state.hyperparam_results

    st.markdown("---")
    st.subheader("📊 实验结果")

    results_sorted = sorted(results, key=lambda x: x['oa'], reverse=True)
    best_idx = results_sorted[0]['index']
    st.session_state.hyperparam_best_idx = best_idx

    df_data = []
    for r in results_sorted:
        row = {'组合编号': r['index'] + 1}
        for k, v in r['params'].items():
            row[k] = str(v)
        row['OA (%)'] = f"{r['oa']*100:.2f}"
        row['Kappa'] = f"{r['kappa']:.4f}"
        row['排名'] = results_sorted.index(r) + 1
        df_data.append(row)

    results_df = pd.DataFrame(df_data)

    def highlight_best(row):
        return ['background-color: #d4edda' if row['排名'] == 1 else '' for _ in row]

    st.dataframe(results_df.style.apply(highlight_best, axis=1), use_container_width=True, height=350)

    best_result = results_sorted[0]
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("🏆 最优OA", f"{best_result['oa']*100:.2f}%")
    col_info2.metric("🏆 最优Kappa", f"{best_result['kappa']:.4f}")
    col_info3.metric("🏆 最优参数组合", f"#{best_result['index'] + 1}")
    st.info(f"🎯 最优参数: {best_result['params']}")

    st.markdown("---")
    if exp_classifier == "SVM (RBF核)" and len(c_list) > 1 and len(gamma_list) > 1:
        st.subheader("🔥 SVM参数热力图")

        oa_matrix = np.zeros((len(c_list), len(gamma_list)))
        kappa_matrix = np.zeros((len(c_list), len(gamma_list)))
        for r in results:
            c_idx = c_list.index(r['params']['C'])
            try:
                g_idx = gamma_list.index(r['params']['gamma'])
            except ValueError:
                continue
            oa_matrix[c_idx, g_idx] = r['oa']
            kappa_matrix[c_idx, g_idx] = r['kappa']

        gamma_labels = [str(g) for g in gamma_list]
        c_labels = [f"{c:.4f}" if isinstance(c, float) else str(c) for c in c_list]

        heatmap_metric = st.radio("热力图指标", ["OA", "Kappa"], horizontal=True, key="heatmap_metric")
        heatmap_data = oa_matrix * 100 if heatmap_metric == "OA" else kappa_matrix
        zmin, zmax = (0, 100) if heatmap_metric == "OA" else (0, 1)

        fig = px.imshow(
            heatmap_data,
            x=gamma_labels, y=c_labels,
            text_auto='.2f' if heatmap_metric == "OA" else '.3f',
            color_continuous_scale='YlOrRd',
            zmin=zmin, zmax=zmax,
            title=f"SVM超参数{heatmap_metric}热力图",
            aspect='auto',
            labels=dict(x="gamma", y="C", color=heatmap_metric)
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_apply1, col_apply2 = st.columns([1, 1])

    with col_apply1:
        selected_idx = st.selectbox(
            "选择要应用的参数组合",
            options=[r['index'] for r in results_sorted],
            format_func=lambda x: f"组合 #{x+1} - OA: {results[x]['oa']*100:.2f}%",
            index=0,
            key="selected_combo"
        )
        selected_params = results[selected_idx]['params']
        st.info(f"已选参数: {selected_params}")

    with col_apply2:
        apply_save_name = st.text_input("结果保存名称", value=f"{exp_classifier.split(' ')[0]}_exp_{pd.Timestamp.now().strftime('%H%M%S')}", key="apply_save_name")
        apply_and_save = st.checkbox("完成后自动保存到对比历史", True, key="apply_save_check")

        if st.button("🚀 用所选参数训练并分类全图", type='primary'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            callback = create_progress_callback(progress_bar, status_text)

            try:
                with st.spinner("正在训练并分类..."):
                    if exp_classifier_key == 'svm':
                        train_params = {
                            'C': [selected_params['C']],
                            'gamma': [selected_params['gamma']],
                            'grid_search': False,
                            'cv': 3
                        }
                        classifier = create_classifier(exp_classifier_key, **train_params)
                        train_X = st.session_state.train_samples.features
                        train_y = st.session_state.train_samples.labels
                        train_info = classifier.fit(train_X, train_y, progress_callback=callback)

                    elif exp_classifier_key == 'random_forest':
                        max_feat = selected_params['max_features']
                        train_params = {
                            'n_estimators': int(selected_params['n_estimators']),
                            'max_depth': int(selected_params['max_depth']),
                            'max_features': max_feat if isinstance(max_feat, str) else float(max_feat)
                        }
                        classifier = create_classifier(exp_classifier_key, **train_params)
                        train_X = st.session_state.train_samples.features
                        train_y = st.session_state.train_samples.labels
                        train_info = classifier.fit(train_X, train_y, progress_callback=callback)

                    elif exp_classifier_key == '1d_cnn':
                        train_params = {
                            'n_epochs': int(selected_params['n_epochs']),
                            'batch_size': int(selected_params['batch_size']),
                            'learning_rate': float(selected_params['learning_rate'])
                        }
                        classifier = create_classifier(exp_classifier_key, **train_params)
                        train_X = st.session_state.train_samples.features
                        train_y = st.session_state.train_samples.labels
                        train_info = classifier.fit(train_X, train_y, progress_callback=callback)

                    elif exp_classifier_key == '3d_cnn':
                        train_params = {
                            'n_epochs': int(selected_params['n_epochs']),
                            'batch_size': int(selected_params['batch_size']),
                            'learning_rate': float(selected_params['learning_rate']),
                            'window_size': int(selected_params['window_size'])
                        }
                        classifier = create_classifier(exp_classifier_key, **train_params)
                        train_X = st.session_state.train_samples.features
                        train_y = st.session_state.train_samples.labels
                        train_locs = st.session_state.train_samples.locations
                        train_info = classifier.fit(
                            train_X, train_y,
                            data_image=st.session_state.data,
                            train_locations=train_locs,
                            progress_callback=callback
                        )

                    else:
                        raise ValueError(f"不支持的分类器: {exp_classifier_key}")

                    st.session_state.classifier = classifier
                    st.session_state.train_info = train_info

                    from src.visualization import classification_to_rgb

                    if exp_classifier_key == '3d_cnn':
                        predictions, _ = classify_image(
                            st.session_state.classifier, st.session_state.features,
                            data_image=st.session_state.data,
                            progress_callback=lambda p, m: callback(0.5 + 0.5 * p, m)
                        )
                    else:
                        predictions, _ = classify_image(
                            st.session_state.classifier, st.session_state.features,
                            progress_callback=lambda p, m: callback(0.5 + 0.5 * p, m)
                        )

                    st.session_state.classification_result = predictions
                    class_rgb, legend = classification_to_rgb(
                        predictions, class_names=st.session_state.samples.class_names
                    )
                    st.session_state.classification_rgb = class_rgb
                    st.session_state.classification_legend = legend

                    if apply_and_save:
                        save_classification_result(
                            apply_save_name,
                            predictions, class_rgb, legend, None
                        )

                    progress_bar.progress(1.0)
                    status_text.text("✅ 完成！")
                    st.success("✅ 训练并分类完成！")

            except Exception as e:
                st.error(f"❌ 失败: {str(e)}")
                progress_bar.progress(0)
                status_text.text("")
                import traceback
                st.code(traceback.format_exc())
