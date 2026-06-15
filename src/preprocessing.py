import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .utils import (
    random_sample_indices, chunk_generator, reshape_for_classifier,
    save_temp_array, load_temp_array, get_temp_file
)


def remove_noisy_bands(data: np.ndarray, band_indices: Optional[List[int]] = None,
                       snr_threshold: float = 0.5, auto_detect: bool = False,
                       chunk_size: int = 1000) -> Tuple[np.ndarray, List[int]]:
    if data.ndim != 3:
        raise ValueError("Data must be 3D array (H, W, B)")

    n_bands = data.shape[2]

    if auto_detect:
        snr = compute_band_snr(data, chunk_size=chunk_size)
        band_indices = np.where(snr < snr_threshold)[0].tolist()
    elif band_indices is None:
        band_indices = []

    keep_indices = [i for i in range(n_bands) if i not in band_indices]

    if len(keep_indices) == n_bands:
        return data, []

    if isinstance(data, np.memmap):
        result = np.empty((data.shape[0], data.shape[1], len(keep_indices)), dtype=data.dtype)
        for chunk, start, end in chunk_generator(data, chunk_size=chunk_size, axis=0):
            result[start:end] = chunk[:, :, keep_indices]
        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
    else:
        result = data[:, :, keep_indices]

    return result, band_indices


def compute_band_snr(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    n_bands = data.shape[2]
    means = np.zeros(n_bands, dtype=np.float64)
    stds = np.zeros(n_bands, dtype=np.float64)
    n_pixels = 0

    for chunk, start, end in chunk_generator(data, chunk_size=chunk_size, axis=0):
        chunk = chunk.astype(np.float64)
        n_chunk = chunk.shape[0] * chunk.shape[1]
        means += np.sum(chunk.reshape(-1, n_bands), axis=0)
        stds += np.sum(np.square(chunk.reshape(-1, n_bands)), axis=0)
        n_pixels += n_chunk

    means /= n_pixels
    stds = np.sqrt(stds / n_pixels - np.square(means))
    snr = means / (stds + 1e-10)
    return snr


def mnf_transform(data: np.ndarray, n_components: int = 10,
                  noise_estimate: str = 'shift_diff',
                  sample_size: int = 100000,
                  chunk_size: int = 1000,
                  progress_callback=None) -> Tuple[np.ndarray, Dict]:
    if data.ndim != 3:
        raise ValueError("Data must be 3D array (H, W, B)")

    H, W, B = data.shape
    n_samples = H * W

    n_sample_use = min(sample_size, n_samples)
    sample_idx = random_sample_indices(n_samples, n_sample_use)

    data_flat = reshape_for_classifier(data).astype(np.float64)
    sample_data = data_flat[sample_idx]

    if progress_callback:
        progress_callback(0.2, "Estimating noise covariance...")

    if noise_estimate == 'shift_diff':
        noise_cov = estimate_noise_covariance(data, sample_idx, chunk_size)
    else:
        noise_cov = np.eye(B)

    if progress_callback:
        progress_callback(0.4, "Computing noise whitening...")

    eigvals_noise, eigvecs_noise = np.linalg.eigh(noise_cov)
    eigvals_noise = np.maximum(eigvals_noise, 1e-10)
    whitening_matrix = eigvecs_noise @ np.diag(1.0 / np.sqrt(eigvals_noise)) @ eigvecs_noise.T

    sample_whitened = sample_data @ whitening_matrix

    if progress_callback:
        progress_callback(0.6, "Computing PCA on whitened data...")

    scaler = StandardScaler()
    sample_whitened_std = scaler.fit_transform(sample_whitened)

    pca = PCA(n_components=min(n_components, B))
    sample_mnf = pca.fit_transform(sample_whitened_std)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    if progress_callback:
        progress_callback(0.8, "Applying MNF transform to full data...")

    n_components_use = np.argmax(cumulative_variance >= 0.99) + 1
    n_components_use = min(n_components, n_components_use)

    transform_matrix = whitening_matrix @ scaler.mean_ * 0
    transform_matrix = whitening_matrix @ (np.eye(B) / scaler.scale_[None, :]) @ pca.components_[:n_components_use].T

    if isinstance(data, np.memmap) or data.nbytes > 500 * 1024 * 1024:
        result = np.empty((H, W, n_components_use), dtype=np.float32)
        for chunk, start, end in chunk_generator(data, chunk_size=chunk_size, axis=0):
            chunk_flat = chunk.reshape(-1, B).astype(np.float64)
            chunk_mnf = (chunk_flat - scaler.mean_) @ transform_matrix
            result[start:end] = chunk_mnf.reshape(chunk.shape[0], W, n_components_use).astype(np.float32)

        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
    else:
        data_flat = data_flat.astype(np.float64)
        data_mnf = (data_flat - scaler.mean_) @ transform_matrix
        result = data_mnf.reshape(H, W, n_components_use).astype(np.float32)

    info = {
        'n_components': n_components_use,
        'explained_variance_ratio': explained_variance_ratio[:n_components_use],
        'cumulative_variance': cumulative_variance[:n_components_use],
        'transform_matrix': transform_matrix,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }

    if progress_callback:
        progress_callback(1.0, "MNF transform complete")

    return result, info


def estimate_noise_covariance(data: np.ndarray, sample_idx: np.ndarray,
                              chunk_size: int = 1000) -> np.ndarray:
    H, W, B = data.shape

    sample_rows = sample_idx // W
    sample_cols = sample_idx % W

    valid_mask = (sample_rows > 0) & (sample_rows < H - 1) & (sample_cols > 0) & (sample_cols < W - 1)
    valid_idx = sample_idx[valid_mask]
    valid_rows = sample_rows[valid_mask]
    valid_cols = sample_cols[valid_mask]

    n_valid = len(valid_idx)
    noise_diff = np.zeros((n_valid * 4, B), dtype=np.float64)

    data_flat = reshape_for_classifier(data)

    noise_diff[:n_valid] = data_flat[valid_idx] - data_flat[(valid_rows + 1) * W + valid_cols]
    noise_diff[n_valid:2*n_valid] = data_flat[valid_idx] - data_flat[(valid_rows - 1) * W + valid_cols]
    noise_diff[2*n_valid:3*n_valid] = data_flat[valid_idx] - data_flat[valid_rows * W + valid_cols + 1]
    noise_diff[3*n_valid:] = data_flat[valid_idx] - data_flat[valid_rows * W + valid_cols - 1]

    noise_diff -= np.mean(noise_diff, axis=0, keepdims=True)
    noise_cov = (noise_diff.T @ noise_diff) / (noise_diff.shape[0] - 1)

    return noise_cov


def pca_transform(data: np.ndarray, n_components: Optional[int] = None,
                  variance_threshold: float = 0.95,
                  chunk_size: int = 1000,
                  sample_size: int = 100000,
                  progress_callback=None) -> Tuple[np.ndarray, Dict]:
    if data.ndim != 3:
        raise ValueError("Data must be 3D array (H, W, B)")

    H, W, B = data.shape
    n_samples = H * W

    n_sample_use = min(sample_size, n_samples)
    sample_idx = random_sample_indices(n_samples, n_sample_use)

    data_flat = reshape_for_classifier(data).astype(np.float64)
    sample_data = data_flat[sample_idx]

    if progress_callback:
        progress_callback(0.2, "Standardizing data...")

    scaler = StandardScaler()
    sample_std = scaler.fit_transform(sample_data)

    if progress_callback:
        progress_callback(0.4, "Computing PCA...")

    pca = PCA(n_components=min(B, 100))
    sample_pca = pca.fit_transform(sample_std)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    if n_components is None:
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_components = min(n_components, B)

    if progress_callback:
        progress_callback(0.7, "Applying PCA transform to full data...")

    transform_matrix = pca.components_[:n_components].T / scaler.scale_[None, :]
    transform_matrix = transform_matrix.astype(np.float64)

    if isinstance(data, np.memmap) or data.nbytes > 500 * 1024 * 1024:
        result = np.empty((H, W, n_components), dtype=np.float32)
        for chunk, start, end in chunk_generator(data, chunk_size=chunk_size, axis=0):
            chunk_flat = chunk.reshape(-1, B).astype(np.float64)
            chunk_pca = (chunk_flat - scaler.mean_) @ transform_matrix
            result[start:end] = chunk_pca.reshape(chunk.shape[0], W, n_components).astype(np.float32)

        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
    else:
        data_flat = data_flat.astype(np.float64)
        data_pca = (data_flat - scaler.mean_) @ transform_matrix
        result = data_pca.reshape(H, W, n_components).astype(np.float32)

    info = {
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_components],
        'cumulative_variance': cumulative_variance[:n_components],
        'transform_matrix': transform_matrix,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }

    if progress_callback:
        progress_callback(1.0, "PCA transform complete")

    return result, info


def savgol_smoothing(data: np.ndarray, window_length: int = 7,
                     polyorder: int = 3, deriv: int = 0,
                     chunk_size: int = 1000,
                     progress_callback=None) -> np.ndarray:
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1

    H, W, B = data.shape

    if isinstance(data, np.memmap) or data.nbytes > 500 * 1024 * 1024:
        result = np.empty_like(data, dtype=np.float32)
        total_chunks = H // chunk_size + 1
        for i, (chunk, start, end) in enumerate(chunk_generator(data, chunk_size=chunk_size, axis=0)):
            chunk_smoothed = savgol_filter(
                chunk.astype(np.float64),
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                axis=2
            )
            result[start:end] = chunk_smoothed.astype(np.float32)

            if progress_callback:
                progress = (i + 1) / total_chunks
                progress_callback(progress, f"Smoothing chunk {i+1}/{total_chunks}")

        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
    else:
        if progress_callback:
            progress_callback(0.5, "Applying Savitzky-Golay filter...")
        result = savgol_filter(
            data.astype(np.float64),
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            axis=2
        ).astype(np.float32)

        if progress_callback:
            progress_callback(1.0, "Smoothing complete")

    return result


def get_cumulative_variance_plot_data(explained_variance_ratio: np.ndarray) -> Dict:
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = len(explained_variance_ratio)

    return {
        'components': np.arange(1, n_components + 1),
        'individual_variance': explained_variance_ratio,
        'cumulative_variance': cumulative,
    }


def preprocessing_pipeline(data: np.ndarray, steps: List[Dict],
                           chunk_size: int = 1000,
                           progress_callback=None) -> Tuple[np.ndarray, Dict]:
    current_data = data
    results = {}
    total_steps = len(steps)

    for i, step in enumerate(steps):
        step_name = step.get('name', step.get('type', 'unknown'))
        step_type = step.get('type', '')

        if progress_callback:
            progress_callback((i + 0.0) / total_steps, f"Step {i+1}/{total_steps}: {step_name}")

        if step_type == 'remove_bands':
            current_data, removed = remove_noisy_bands(
                current_data,
                band_indices=step.get('band_indices'),
                snr_threshold=step.get('snr_threshold', 0.5),
                auto_detect=step.get('auto_detect', False),
                chunk_size=chunk_size
            )
            results['removed_bands'] = removed

        elif step_type == 'mnf':
            current_data, mnf_info = mnf_transform(
                current_data,
                n_components=step.get('n_components', 10),
                noise_estimate=step.get('noise_estimate', 'shift_diff'),
                sample_size=step.get('sample_size', 100000),
                chunk_size=chunk_size,
                progress_callback=lambda p, m: progress_callback(
                    (i + 0.8 * p) / total_steps,
                    f"Step {i+1}/{total_steps}: {m}"
                )
            )
            results['mnf'] = mnf_info

        elif step_type == 'pca':
            current_data, pca_info = pca_transform(
                current_data,
                n_components=step.get('n_components'),
                variance_threshold=step.get('variance_threshold', 0.95),
                chunk_size=chunk_size,
                sample_size=step.get('sample_size', 100000),
                progress_callback=lambda p, m: progress_callback(
                    (i + 0.8 * p) / total_steps,
                    f"Step {i+1}/{total_steps}: {m}"
                )
            )
            results['pca'] = pca_info

        elif step_type == 'savgol':
            current_data = savgol_smoothing(
                current_data,
                window_length=step.get('window_length', 7),
                polyorder=step.get('polyorder', 3),
                deriv=step.get('deriv', 0),
                chunk_size=chunk_size,
                progress_callback=lambda p, m: progress_callback(
                    (i + 0.8 * p) / total_steps,
                    f"Step {i+1}/{total_steps}: {m}"
                )
            )
            results['savgol'] = True

    if progress_callback:
        progress_callback(1.0, "Preprocessing pipeline complete")

    return current_data, results
