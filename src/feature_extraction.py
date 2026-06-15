import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from scipy import ndimage
from scipy.ndimage import morphology
from scipy.spatial import ConvexHull
from tqdm import tqdm
import cv2
from .utils import chunk_generator, save_temp_array, load_temp_array, reshape_for_classifier


def continuum_removal(spectrum: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
    if wavelengths is None:
        wavelengths = np.arange(len(spectrum))

    spectrum = spectrum.astype(np.float64)
    n = len(spectrum)

    points = np.column_stack([wavelengths, spectrum])
    hull = ConvexHull(points)

    hull_vertices = hull.vertices
    hull_vertices = np.sort(hull_vertices)

    hull_spectrum = np.interp(wavelengths, wavelengths[hull_vertices], spectrum[hull_vertices])

    hull_spectrum = np.maximum(hull_spectrum, spectrum)

    cr_spectrum = spectrum / (hull_spectrum + 1e-10)

    return cr_spectrum


def spectral_derivatives(spectrum: np.ndarray, order: int = 1,
                         window_length: int = 5, polyorder: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter

    if window_length % 2 == 0:
        window_length += 1

    deriv = savgol_filter(
        spectrum.astype(np.float64),
        window_length=window_length,
        polyorder=polyorder,
        deriv=order,
        axis=0
    )

    return deriv


def detect_absorption_peaks(spectrum: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                            threshold: float = 0.05, min_depth: float = 0.02) -> Dict:
    if wavelengths is None:
        wavelengths = np.arange(len(spectrum))

    cr_spectrum = continuum_removal(spectrum, wavelengths)

    from scipy.signal import find_peaks

    valleys, _ = find_peaks(1 - cr_spectrum, height=min_depth, distance=3)

    peaks_info = []
    for idx in valleys:
        left_idx = idx
        while left_idx > 0 and cr_spectrum[left_idx - 1] > cr_spectrum[left_idx]:
            left_idx -= 1

        right_idx = idx
        while right_idx < len(cr_spectrum) - 1 and cr_spectrum[right_idx + 1] > cr_spectrum[right_idx]:
            right_idx += 1

        if right_idx > left_idx + 1:
            depth = 1 - cr_spectrum[idx]
            if depth >= threshold:
                peaks_info.append({
                    'wavelength': wavelengths[idx],
                    'depth': depth,
                    'width': wavelengths[right_idx] - wavelengths[left_idx],
                    'left_edge': wavelengths[left_idx],
                    'right_edge': wavelengths[right_idx],
                    'index': idx
                })

    return {
        'num_peaks': len(peaks_info),
        'peaks': peaks_info,
        'continuum_removed': cr_spectrum
    }


def extract_spectral_features(data: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                              features: List[str] = None,
                              chunk_size: int = 1000,
                              progress_callback=None) -> Tuple[np.ndarray, Dict]:
    if features is None:
        features = ['continuum_removal', 'first_derivative', 'second_derivative', 'absorption_peaks']

    H, W, B = data.shape

    if wavelengths is None:
        wavelengths = np.arange(B)

    feature_list = []
    feature_names = []

    if 'continuum_removal' in features:
        feature_names.extend([f'CR_{i}' for i in range(B)])

    if 'first_derivative' in features:
        feature_names.extend([f'1stDeriv_{i}' for i in range(B)])

    if 'second_derivative' in features:
        feature_names.extend([f'2ndDeriv_{i}' for i in range(B)])

    if 'absorption_peaks' in features:
        feature_names.extend(['num_peaks', 'max_peak_depth', 'mean_peak_depth'])

    n_features = len(feature_names)

    if isinstance(data, np.memmap) or data.nbytes > 500 * 1024 * 1024:
        result = np.empty((H, W, n_features), dtype=np.float32)
        total_chunks = H // chunk_size + 1

        for i, (chunk, start, end) in enumerate(chunk_generator(data, chunk_size=chunk_size, axis=0)):
            chunk_features = _extract_spectral_chunk(chunk, wavelengths, features)
            result[start:end] = chunk_features.astype(np.float32)

            if progress_callback:
                progress = (i + 1) / total_chunks
                progress_callback(progress, f"Extracting spectral features chunk {i+1}/{total_chunks}")

        tmp_path = save_temp_array(result)
        result = load_temp_array(tmp_path, mmap=True)
    else:
        if progress_callback:
            progress_callback(0.5, "Extracting spectral features...")
        result = _extract_spectral_chunk(data, wavelengths, features).astype(np.float32)

        if progress_callback:
            progress_callback(1.0, "Spectral feature extraction complete")

    info = {
        'feature_names': feature_names,
        'n_features': n_features,
        'features': features
    }

    return result, info


def _extract_spectral_chunk(chunk: np.ndarray, wavelengths: np.ndarray,
                            features: List[str]) -> np.ndarray:
    H, W, B = chunk.shape
    chunk_flat = chunk.reshape(-1, B).astype(np.float64)
    n_pixels = chunk_flat.shape[0]

    all_features = []

    if 'continuum_removal' in features:
        cr_features = np.apply_along_axis(
            lambda x: continuum_removal(x, wavelengths), 1, chunk_flat
        )
        all_features.append(cr_features)

    if 'first_derivative' in features:
        deriv1 = np.apply_along_axis(
            lambda x: spectral_derivatives(x, order=1), 1, chunk_flat
        )
        all_features.append(deriv1)

    if 'second_derivative' in features:
        deriv2 = np.apply_along_axis(
            lambda x: spectral_derivatives(x, order=2), 1, chunk_flat
        )
        all_features.append(deriv2)

    if 'absorption_peaks' in features:
        peak_features = np.zeros((n_pixels, 3), dtype=np.float64)
        for i in range(n_pixels):
            peaks = detect_absorption_peaks(chunk_flat[i], wavelengths)
            depths = [p['depth'] for p in peaks['peaks']]
            peak_features[i, 0] = peaks['num_peaks']
            peak_features[i, 1] = max(depths) if depths else 0
            peak_features[i, 2] = np.mean(depths) if depths else 0
        all_features.append(peak_features)

    combined = np.concatenate(all_features, axis=1)
    return combined.reshape(H, W, -1)


def morphological_profile(image: np.ndarray, scales: List[int] = None,
                          operations: List[str] = None) -> np.ndarray:
    if scales is None:
        scales = [3, 5, 7, 9, 11]

    if operations is None:
        operations = ['opening', 'closing']

    if image.ndim == 3:
        if image.shape[2] > 3:
            image = np.mean(image, axis=2)
        else:
            image = image[:, :, 0]

    features = []

    for scale in scales:
        selem = morphology.disk(scale // 2)

        for op in operations:
            if op == 'opening':
                result = morphology.grey_opening(image, footprint=selem)
            elif op == 'closing':
                result = morphology.grey_closing(image, footprint=selem)
            elif op == 'dilation':
                result = morphology.grey_dilation(image, footprint=selem)
            elif op == 'erosion':
                result = morphology.grey_erosion(image, footprint=selem)
            else:
                continue

            features.append(result)

    return np.stack(features, axis=2)


def gabor_features(image: np.ndarray, frequencies: List[float] = None,
                   angles: List[float] = None) -> np.ndarray:
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3, 0.4]

    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    if image.ndim == 3:
        if image.shape[2] > 3:
            image = np.mean(image, axis=2)
        else:
            image = image[:, :, 0]

    image = image.astype(np.float32)
    features = []

    for freq in frequencies:
        for theta in angles:
            kernel = cv2.getGaborKernel(
                (31, 31), 4, theta, freq, 0.5, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            features.append(filtered)

    return np.stack(features, axis=2)


def extract_spatial_features(data: np.ndarray,
                             features: List[str] = None,
                             mp_scales: List[int] = None,
                             gabor_frequencies: List[float] = None,
                             gabor_angles: List[float] = None,
                             chunk_size: int = 1000,
                             progress_callback=None) -> Tuple[np.ndarray, Dict]:
    if features is None:
        features = ['morphological_profile', 'gabor']

    H, W, B = data.shape

    feature_names = []

    if 'morphological_profile' in features:
        scales = mp_scales if mp_scales else [3, 5, 7, 9, 11]
        ops = ['opening', 'closing']
        feature_names.extend([f'MP_{op}_s{s}' for s in scales for op in ops])

    if 'gabor' in features:
        freqs = gabor_frequencies if gabor_frequencies else [0.1, 0.2, 0.3, 0.4]
        angles = gabor_angles if gabor_angles else [0, np.pi/4, np.pi/2, 3*np.pi/4]
        feature_names.extend([f'Gabor_f{f:.2f}_a{a:.2f}' for f in freqs for a in angles])

    n_features = len(feature_names)
    result = np.empty((H, W, n_features), dtype=np.float32)

    if progress_callback:
        progress_callback(0.2, "Computing morphological profiles...")

    if 'morphological_profile' in features:
        mp = morphological_profile(data, mp_scales, ['opening', 'closing'])
        n_mp = mp.shape[2]
        result[:, :, :n_mp] = mp.astype(np.float32)
        current_idx = n_mp
    else:
        current_idx = 0

    if progress_callback:
        progress_callback(0.6, "Computing Gabor features...")

    if 'gabor' in features:
        gabor = gabor_features(data, gabor_frequencies, gabor_angles)
        result[:, :, current_idx:] = gabor.astype(np.float32)

    if progress_callback:
        progress_callback(1.0, "Spatial feature extraction complete")

    info = {
        'feature_names': feature_names,
        'n_features': n_features,
        'features': features
    }

    return result, info


def extract_features(data: np.ndarray,
                     feature_type: str = 'spectral',
                     wavelengths: Optional[np.ndarray] = None,
                     spectral_features: List[str] = None,
                     spatial_features: List[str] = None,
                     mp_scales: List[int] = None,
                     gabor_frequencies: List[float] = None,
                     chunk_size: int = 1000,
                     progress_callback=None) -> Tuple[np.ndarray, Dict]:
    if feature_type == 'spectral':
        return extract_spectral_features(
            data, wavelengths, spectral_features, chunk_size, progress_callback
        )
    elif feature_type == 'spatial':
        return extract_spatial_features(
            data, spatial_features, mp_scales, gabor_frequencies, chunk_size, progress_callback
        )
    elif feature_type == 'fused':
        if progress_callback:
            progress_callback(0.0, "Extracting spectral features...")

        spec_feat, spec_info = extract_spectral_features(
            data, wavelengths, spectral_features, chunk_size,
            lambda p, m: progress_callback(0.4 * p, m)
        )

        if progress_callback:
            progress_callback(0.5, "Extracting spatial features...")

        spat_feat, spat_info = extract_spatial_features(
            data, spatial_features, mp_scales, gabor_frequencies, chunk_size,
            lambda p, m: progress_callback(0.5 + 0.4 * p, m)
        )

        if progress_callback:
            progress_callback(0.9, "Fusing features...")

        spec_flat = reshape_for_classifier(spec_feat)
        spat_flat = reshape_for_classifier(spat_feat)

        fused_flat = np.concatenate([spec_flat, spat_flat], axis=1)

        H, W = data.shape[:2]
        fused = fused_flat.reshape(H, W, -1)

        if isinstance(data, np.memmap) or fused.nbytes > 500 * 1024 * 1024:
            tmp_path = save_temp_array(fused)
            fused = load_temp_array(tmp_path, mmap=True)

        info = {
            'spectral_features': spec_info,
            'spatial_features': spat_info,
            'feature_names': spec_info['feature_names'] + spat_info['feature_names'],
            'n_features': spec_info['n_features'] + spat_info['n_features'],
            'feature_type': 'fused'
        }

        if progress_callback:
            progress_callback(1.0, "Feature fusion complete")

        return fused, info
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
