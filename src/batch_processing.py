import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Union, Any
from dataclasses import dataclass
from tqdm import tqdm
from .classification import HSIClassifier
from .utils import chunk_generator, save_temp_array, load_temp_array, reshape_for_classifier
from .preprocessing import preprocessing_pipeline
from .feature_extraction import extract_features
from .visualization import classification_to_rgb
from .data_io import load_envi_data, save_envi_data
from .classification import classify_image


@dataclass
class BatchJob:
    hdr_path: str
    dat_path: str
    output_dir: str
    output_name: str
    processed: bool = False
    success: bool = False
    error: Optional[str] = None


def process_in_chunks(data: np.ndarray,
                  classifier: HSIClassifier,
                  chunk_size: int = 500,
                  overlap: int = 32,
                  progress_callback=None) -> np.ndarray:
    H, W, B = data.shape
    predictions = np.zeros((H, W), dtype=np.int32)
    count_map = np.zeros((H, W), dtype=np.int32)

    stride = chunk_size - overlap

    total_blocks = ((H + stride - 1) // stride) * ((W + stride - 1) // stride)
    block_idx = 0

    for r_start in range(0, H, stride):
        r_end = min(r_start + chunk_size, H)
        r_pad_start = max(0, r_start - overlap)
        r_pad_end = min(H, r_end + overlap)

        for c_start in range(0, W, stride):
            c_end = min(c_start + chunk_size, W)
            c_pad_start = max(0, c_start - overlap)
            c_pad_end = min(W, c_end + overlap)

            block = data[r_pad_start:r_pad_end, c_pad_start:c_pad_end]

            block_pred = _predict_block(
                block, classifier,
                (r_pad_start, r_pad_end),
                (c_pad_start, c_pad_end)
            )

            r_local_start = r_start - r_pad_start
            r_local_end = r_end - r_pad_start
            c_local_start = c_start - c_pad_start
            c_local_end = c_end - c_pad_start

            pred_inner = block_pred[r_local_start:r_local_end, c_local_start:c_local_end]

            predictions[r_start:r_end, c_start:c_end] += pred_inner
            count_map[r_start:r_end, c_start:c_end] += 1

            block_idx += 1
            if progress_callback:
                progress = block_idx / total_blocks
                progress_callback(progress, f"Processing block {block_idx}/{total_blocks}")

    predictions = predictions / np.maximum(count_map, 1)
    predictions = predictions.astype(np.int32)

    return predictions


def _predict_block(block: np.ndarray, classifier: HSIClassifier,
               r_range: Tuple[int, int], c_range: Tuple[int, int]) -> np.ndarray:
    pred, _ = classify_image(classifier, block, chunk_size=1000)
    return pred


def process_single_image(data_path: str, hdr_path: str,
                         preprocess_steps: List[Dict],
                         feature_config: Dict,
                         classifier: HSIClassifier,
                         output_dir: str,
                         output_name: str,
                         wavelengths: Optional[np.ndarray] = None,
                         chunk_size: int = 500,
                         overlap: int = 32,
                         progress_callback=None) -> Dict:
    try:
        if progress_callback:
            progress_callback(0.0, "Loading data...")

        data, header = load_envi_data(hdr_path, data_path, mmap=True)

        if progress_callback:
            progress_callback(0.2, "Preprocessing...")

        preprocessed_data, preprocess_results = preprocessing_pipeline(
            data, preprocess_steps,
            progress_callback=lambda p, m: progress_callback(0.2 + 0.3 * p, m)
        )

        if progress_callback:
            progress_callback(0.5, "Extracting features...")

        features, feature_info = extract_features(
            preprocessed_data,
            feature_type=feature_config.get('feature_type', 'spectral'),
            wavelengths=wavelengths,
            spectral_features=feature_config.get('spectral_features'),
            spatial_features=feature_config.get('spatial_features'),
            mp_scales=feature_config.get('mp_scales'),
            gabor_frequencies=feature_config.get('gabor_frequencies'),
            chunk_size=1000,
            progress_callback=lambda p, m: progress_callback(0.5 + 0.2 * p, m)
        )

        if progress_callback:
            progress_callback(0.7, "Classifying...")

        if isinstance(classifier, HSIClassifier) and not hasattr(classifier, 'window_size'):
            predictions = _predict_block(features, classifier, (0, features.shape[0]), (0, features.shape[1]))
        else:
            predictions = process_in_chunks(
                features, classifier,
                chunk_size=chunk_size,
                overlap=overlap,
                progress_callback=lambda p, m: progress_callback(0.7 + 0.2 * p, m)
            )

        if progress_callback:
            progress_callback(0.9, "Saving results...")

        class_rgb, legend = classification_to_rgb(predictions)

        base_name = os.path.join(output_dir, output_name)

        pred_hdr = base_name + '_classification.hdr'
        pred_dat = base_name + '_classification'
        save_envi_data(predictions[:, :, np.newaxis], pred_hdr, pred_dat)

        from PIL import Image
        rgb_img = (class_rgb * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(base_name + '_classification.png')

        if progress_callback:
            progress_callback(1.0, "Processing complete")

        return {
            'success': True,
            'predictions': predictions,
            'classification_rgb': class_rgb,
            'legend': legend,
            'preprocess_results': preprocess_results,
            'feature_info': feature_info,
            'output_files': {
                'hdr': pred_hdr,
                'dat': pred_dat,
                'png': base_name + '_classification.png'
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def batch_process(jobs: List[BatchJob],
                  preprocess_steps: List[Dict],
                  feature_config: Dict,
                  classifier: HSIClassifier,
                  output_dir: str,
                  wavelengths: Optional[np.ndarray] = None,
                  chunk_size: int = 500,
                  overlap: int = 32,
                  job_progress_callback=None,
                  overall_progress_callback=None) -> List[Dict]:
    results = []
    n_jobs = len(jobs)

    for i, job in enumerate(jobs):
        if overall_progress_callback:
            overall_progress_callback(i / n_jobs, f"Processing job {i+1}/{n_jobs}: {job.output_name}")

        result = process_single_image(
            job.dat_path, job.hdr_path,
            preprocess_steps,
            feature_config,
            classifier,
            job.output_dir,
            job.output_name,
            wavelengths=wavelengths,
            chunk_size=chunk_size,
            overlap=overlap,
            progress_callback=job_progress_callback
        )

        job.processed = True
        job.success = result.get('success', False)
        job.error = result.get('error')

        results.append(result)

    if overall_progress_callback:
        overall_progress_callback(1.0, "Batch processing complete")

    return results


def create_batch_jobs(input_files: List[Tuple[str, str]], output_dir: str) -> List[BatchJob]:
    jobs = []
    for i, (hdr_path, dat_path) in enumerate(input_files):
        base_name = os.path.splitext(os.path.basename(hdr_path))[0]
        job = BatchJob(
            hdr_path=hdr_path,
            dat_path=dat_path,
            output_dir=output_dir,
            output_name=f"batch_{i}_{base_name}"
        )
        jobs.append(job)
    return jobs
