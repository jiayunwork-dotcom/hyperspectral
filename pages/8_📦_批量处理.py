import streamlit as st
import os

from state import init_session_state, save_uploaded_file, create_progress_callback, render_sidebar_info
from src.batch_processing import create_batch_jobs, batch_process


st.set_page_config(page_title="批量处理", page_icon="📦", layout="wide")
init_session_state()
render_sidebar_info()

st.header("📦 批量处理")

if st.session_state.classifier is None:
    st.warning("⚠️ 请先训练好分类模型再进行批量处理")
    st.stop()

st.markdown("上传多景影像，使用已训练的模型和参数批量执行分类。")

st.subheader("📁 上传批量数据")
num_files = st.number_input("要处理的影像数量", 1, 10, 1)

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
    chunk_size = st.slider("处理块大小", 200, 1000, 500, 50)
    overlap = st.slider("块重叠大小", 16, 128, 32, 16)
with col2:
    output_dir = st.text_input("输出目录", "./batch_outputs")

if st.button("🚀 开始批量处理", type='primary'):
    if len(batch_files) < num_files:
        st.warning(f"⚠️ 请上传 {num_files} 组完整的数据文件")
        st.stop()

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
            feature_config = {'feature_type': 'spectral', 'spectral_features': ['continuum_removal', 'first_derivative'], 'spatial_features': None}

            job_progress = st.progress(0)
            job_status = st.empty()
            job_callback = create_progress_callback(job_progress, job_status)

            results = batch_process(
                jobs, preprocess_steps, feature_config,
                st.session_state.classifier, output_dir,
                wavelengths=st.session_state.wavelengths,
                chunk_size=chunk_size, overlap=overlap,
                job_progress_callback=job_callback,
                overall_progress_callback=overall_callback
            )

            st.success("✅ 批量处理完成！")
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
