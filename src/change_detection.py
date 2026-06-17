import numpy as np
import json
import cv2
from typing import Tuple, Dict, List, Optional
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from .utils import reshape_for_classifier, chunk_generator


def align_images(img_a: np.ndarray, img_b: np.ndarray,
               align_mode: str = 'crop_min') -> Tuple[np.ndarray, np.ndarray, Dict]:
    if img_a.ndim != 3 or img_b.ndim != 3:
        raise ValueError("Images must be 3D arrays (H, W, B)")

    Ha, Wa, Ba = img_a.shape
    Hb, Wb, Bb = img_b.shape

    info = {
        'original_shape_a': (Ha, Wa, Ba),
        'original_shape_b': (Hb, Wb, Bb),
        'bands_match': Ba == Bb,
        'spatial_match': (Ha == Hb) and (Wa == Wb),
    }

    B_common = min(Ba, Bb)

    if align_mode == 'crop_min':
        H = min(Ha, Hb)
        W = min(Wa, Wb)
        img_a_aligned = img_a[:H, :W, :B_common]
        img_b_aligned = img_b[:H, :W, :B_common]
        info['aligned_shape'] = (H, W, B_common)
        info['align_mode'] = 'crop_min'
    elif align_mode == 'center_crop':
        H = min(Ha, Hb)
        W = min(Wa, Wb)
        start_a_h = (Ha - H) // 2
        start_a_w = (Wa - W) // 2
        start_b_h = (Hb - H) // 2
        start_b_w = (Wb - W) // 2
        img_a_aligned = img_a[start_a_h:start_a_h+H, start_a_w:start_a_w+W, :B_common]
        img_b_aligned = img_b[start_b_h:start_b_h+H, start_b_w:start_b_w+W, :B_common]
        info['aligned_shape'] = (H, W, B_common)
        info['align_mode'] = 'center_crop'
    else:
        raise ValueError(f"Unknown align mode: {align_mode}")

    return img_a_aligned, img_b_aligned, info


def compute_sad(img_a: np.ndarray, img_b: np.ndarray,
               chunk_size: int = 1000,
               progress_callback=None) -> np.ndarray:
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

    H, W, B = img_a.shape
    sad_map = np.zeros((H, W), dtype=np.float32)
    total_chunks = (H + chunk_size - 1) // chunk_size
    chunk_idx = 0

    for chunk_a, start, end in chunk_generator(img_a, chunk_size=chunk_size, axis=0):
        chunk_b = img_b[start:end]
        chunk_h = chunk_a.astype(np.float64)
        chunk_h2 = chunk_b.astype(np.float64)

        dot = np.sum(chunk_h * chunk_h2, axis=-1)
        norm_a = np.linalg.norm(chunk_h, axis=-1)
        norm_b = np.linalg.norm(chunk_h2, axis=-1)

        cos_angle = dot / (norm_a * norm_b + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angle = np.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

        sad_map[start:end] = angle.astype(np.float32)

        chunk_idx += 1
        if progress_callback:
            progress = (chunk_idx / total_chunks) * 0.8
            progress_callback(progress, f"计算SAD: 行 {start}-{end}/{H}")

    return sad_map


def sad_change_detection(img_a: np.ndarray, img_b: np.ndarray,
                        threshold: float,
                        chunk_size: int = 1000,
                        progress_callback=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if progress_callback:
        progress_callback(0.0, "开始计算SAD...")

    sad_map = compute_sad(img_a, img_b, chunk_size=chunk_size, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(0.85, "生成变化掩码...")

    change_mask = sad_map > threshold

    if progress_callback:
        progress_callback(0.95, "计算统计信息...")

    stats = {
        'method': 'SAD',
        'threshold': threshold,
        'sad_mean': float(np.mean(sad_map)),
        'sad_std': float(np.std(sad_map)),
        'sad_min': float(np.min(sad_map)),
        'sad_max': float(np.max(sad_map)),
        'change_pixels': int(np.sum(change_mask)),
        'total_pixels': int(change_mask.size),
        'change_ratio': float(np.sum(change_mask) / change_mask.size),
    }

    if progress_callback:
        progress_callback(1.0, "SAD变化检测完成")

    return change_mask, sad_map, stats


def compute_cva(img_a: np.ndarray, img_b: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

    H, W, B = img_a.shape
    magnitude = np.zeros((H, W), dtype=np.float32)
    total_chunks = (H + chunk_size - 1) // chunk_size
    chunk_idx = 0

    for chunk_a, start, end in chunk_generator(img_a, chunk_size=chunk_size, axis=0):
        chunk_b = img_b[start:end]
        diff = chunk_a.astype(np.float64) - chunk_b.astype(np.float64)
        mag = np.linalg.norm(diff, axis=-1)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        magnitude[start:end] = mag.astype(np.float32)

        chunk_idx += 1
        if progress_callback:
            progress = (chunk_idx / total_chunks) * 0.8
            progress_callback(progress, f"计算CVA: 行 {start}-{end}/{H}")

    return magnitude


def cva_change_detection(img_a: np.ndarray, img_b: np.ndarray,
                         threshold: Optional[float] = None,
                         threshold_method: str = 'manual',
                         percentile: float = 95.0,
                         chunk_size: int = 1000,
                         progress_callback=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if progress_callback:
        progress_callback(0.0, "开始计算CVA...")

    magnitude = compute_cva(img_a, img_b, chunk_size=chunk_size, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(0.82, "确定阈值...")

    if threshold_method == 'percentile':
        threshold = float(np.percentile(magnitude, percentile))
    elif threshold is None:
        threshold = float(np.mean(magnitude) + 2 * np.std(magnitude))

    if progress_callback:
        progress_callback(0.88, "生成变化掩码...")

    change_mask = magnitude > threshold

    if progress_callback:
        progress_callback(0.95, "计算统计信息...")

    stats = {
        'method': 'CVA',
        'threshold_method': threshold_method,
        'threshold': threshold,
        'magnitude_mean': float(np.mean(magnitude)),
        'magnitude_std': float(np.std(magnitude)),
        'magnitude_min': float(np.min(magnitude)),
        'magnitude_max': float(np.max(magnitude)),
        'change_pixels': int(np.sum(change_mask)),
        'total_pixels': int(change_mask.size),
        'change_ratio': float(np.sum(change_mask) / change_mask.size),
    }

    if progress_callback:
        progress_callback(1.0, "CVA变化检测完成")

    return change_mask, magnitude, stats


def otsu_threshold(image: np.ndarray,
                   num_bins: int = 256) -> float:
    flat = image.ravel()

    hist, bin_edges = np.histogram(flat, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = flat.size
    sum_total = np.sum(hist * bin_centers)

    sum_back = 0.0
    w_back = 0
    w_fore = 0
    var_max = 0.0
    threshold = bin_centers[0]

    for i in range(num_bins):
        w_back += hist[i]
        if w_back == 0:
            continue

        w_fore = total - w_back
        if w_fore == 0:
            break

        sum_back += hist[i] * bin_centers[i]
        mean_back = sum_back / w_back
        mean_fore = (sum_total - sum_back) / w_fore

        var_between = w_back * w_fore * (mean_back - mean_fore) ** 2

        if var_between > var_max:
            var_max = var_between
            threshold = bin_centers[i]

    return float(threshold)


def pca_change_detection(img_a: np.ndarray, img_b: np.ndarray,
                        variance_ratio: float = 0.95,
                        chunk_size: int = 1000,
                        progress_callback=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

    H, W, B = img_a.shape

    if progress_callback:
        progress_callback(0.0, "计算差值影像...")

    diff = (img_a.astype(np.float64) - img_b.astype(np.float64))
    diff_flat = reshape_for_classifier(diff)

    if progress_callback:
        progress_callback(0.15, "执行PCA分析...")

    pca = PCA()
    pca.fit(diff_flat)

    if progress_callback:
        progress_callback(0.35, "确定主成分数...")

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_ratio) + 1
    n_components = min(n_components, B)

    if progress_callback:
        progress_callback(0.5, f"使用 {n_components} 个主成分变换...")

    pca_n = PCA(n_components=n_components)
    pca_features = pca_n.fit_transform(diff_flat)

    if progress_callback:
        progress_callback(0.7, "计算变化强度...")

    magnitude = np.linalg.norm(pca_features, axis=1)
    magnitude = magnitude.reshape(H, W).astype(np.float32)
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    if progress_callback:
        progress_callback(0.8, "OTSU自动确定阈值...")

    threshold = otsu_threshold(magnitude)

    if progress_callback:
        progress_callback(0.88, "生成变化掩码...")

    change_mask = magnitude > threshold

    if progress_callback:
        progress_callback(0.95, "计算统计信息...")

    stats = {
        'method': 'PCA',
        'variance_ratio': variance_ratio,
        'n_components': int(n_components),
        'threshold': threshold,
        'explained_variance_ratio': pca_n.explained_variance_ratio_,
        'cumulative_variance': float(cumulative_variance[n_components - 1]),
        'magnitude_mean': float(np.mean(magnitude)),
        'magnitude_std': float(np.std(magnitude)),
        'magnitude_min': float(np.min(magnitude)),
        'magnitude_max': float(np.max(magnitude)),
        'change_pixels': int(np.sum(change_mask)),
        'total_pixels': int(change_mask.size),
        'change_ratio': float(np.sum(change_mask) / change_mask.size),
    }

    if progress_callback:
        progress_callback(1.0, "PCA变化检测完成")

    return change_mask, magnitude, stats


def create_change_visualization(change_mask: np.ndarray,
                           intensity_map: np.ndarray,
                           background: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    H, W = change_mask.shape

    if background is None:
        bg = np.ones((H, W, 3), dtype=np.float32) * 0.5
    else:
        bg = background.copy()
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=-1)
        if bg.dtype != np.float32:
            bg = bg.astype(np.float32)
        if bg.max() > 1.0:
            bg = bg / 255.0

    binary_vis = bg.copy()
    binary_vis[change_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    unchanged = ~change_mask
    gray_vals = np.mean(bg[unchanged], axis=-1)
    gray_vals = gray_vals * 0.6
    binary_vis[unchanged] = np.stack([gray_vals, gray_vals, gray_vals], axis=-1)

    from matplotlib import cm
    intensity_norm = (intensity_map - intensity_map.min()) / (intensity_map.max() - intensity_map.min() + 1e-10)
    intensity_norm = np.clip(intensity_norm, 0, 1)
    cmap = cm.get_cmap('hot')
    heatmap = cmap(intensity_norm)[:, :, :3].astype(np.float32)

    heat_vis = bg.copy()
    heat_vis = heat_vis * 0.3 + heatmap * 0.7

    return binary_vis, heat_vis


def compute_transition_matrix(class_a: np.ndarray,
                             class_b: np.ndarray,
                             class_names: Dict[int, str],
                             change_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int], Dict]:
    if class_a.shape != class_b.shape:
        raise ValueError("Classification maps must have the same shape")

    if change_mask is not None:
        valid_a = class_a[change_mask]
        valid_b = class_b[change_mask]
    else:
        valid_a = class_a.ravel()
        valid_b = class_b.ravel()

    classes = sorted(np.unique(np.concatenate([valid_a, valid_b])).tolist())
    n_classes = len(classes)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    transition_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(len(valid_a)):
        idx_a = class_to_idx.get(valid_a[i])
        idx_b = class_to_idx.get(valid_b[i])
        if idx_a is not None and idx_b is not None:
            transition_matrix[idx_a, idx_b] += 1

    stats = {
        'total_changed_pixels': int(len(valid_a)),
        'classes': classes,
        'class_names': [class_names.get(c, f'Class {c}') for c in classes],
        'transition_matrix': transition_matrix.tolist(),
    }

    return transition_matrix, classes, stats


def prepare_sankey_data(transition_matrix: np.ndarray,
                    classes: List[int],
                    class_names: Dict[int, str]) -> Dict:
    n = len(classes)
    labels = []
    source = []
    target = []
    value = []

    for i in range(n):
        labels.append(f"{class_names.get(classes[i], f'Class {classes[i]}')} (时相A)")
    for i in range(n):
        labels.append(f"{class_names.get(classes[i], f'Class {classes[i]}')} (时相B)")

    for i in range(n):
        for j in range(n):
            if transition_matrix[i, j] > 0:
                source.append(i)
                target.append(n + j)
                value.append(int(transition_matrix[i, j]))

    return {
        'labels': labels,
        'source': source,
        'target': target,
        'value': value,
        'n_classes': n,
    }


def compute_spectral_difference(spectrum_a: np.ndarray,
                          spectrum_b: np.ndarray,
                          wavelengths: Optional[np.ndarray] = None) -> Dict:
    if len(spectrum_a) != len(spectrum_b):
        raise ValueError("Spectra must have the same length")

    diff = spectrum_a - spectrum_b
    abs_diff = np.abs(diff)
    max_diff_idx = int(np.argmax(abs_diff))

    norm_a = np.linalg.norm(spectrum_a)
    norm_b = np.linalg.norm(spectrum_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        sad = 0.0
    else:
        cos_angle = np.dot(spectrum_a, spectrum_b) / (norm_a * norm_b + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        sad = np.arccos(cos_angle)
        if np.isnan(sad) or np.isinf(sad):
            sad = 0.0
        if sad < 1e-5:
            sad = 0.0

    sad = float(np.clip(sad, 0, np.pi))

    euclidean = float(np.linalg.norm(diff))
    if np.isnan(euclidean) or np.isinf(euclidean):
        euclidean = 0.0

    stats = {
        'max_diff_band_index': max_diff_idx,
        'max_diff_value': float(abs_diff[max_diff_idx]),
        'mean_diff': float(np.mean(abs_diff)),
        'std_diff': float(np.std(abs_diff)),
        'sad': sad,
        'euclidean_distance': euclidean,
        'difference': diff,
        'absolute_difference': abs_diff,
    }

    if wavelengths is not None and len(wavelengths) == len(spectrum_a):
        stats['max_diff_wavelength'] = float(wavelengths[max_diff_idx])

    return stats


def average_spectrum_in_region(data: np.ndarray,
                          region_mask: np.ndarray) -> np.ndarray:
    if data.shape[:2] != region_mask.shape:
        raise ValueError("Data and mask must have same spatial dimensions")

    if not np.any(region_mask):
        return np.zeros(data.shape[2], dtype=np.float32)

    masked_data = data[region_mask]
    avg_spectrum = np.mean(masked_data, axis=0)

    return avg_spectrum.astype(np.float32)


def mask_to_polygons(change_mask: np.ndarray,
                    min_area_pixels: int = 10,
                    simplify_tolerance: float = 1.0) -> List[Dict]:
    mask_uint8 = (change_mask.astype(np.uint8) * 255)

    contours, hierarchy = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_pixels:
            continue

        epsilon = simplify_tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            coords = approx.reshape(-1, 2).tolist()
            coords.append(coords[0])
            polygons.append({
                'type': 'Polygon',
                'coordinates': [coords],
                'area_pixels': int(area),
            })

    return polygons


def export_change_to_geojson(change_mask: np.ndarray,
                            intensity_map: np.ndarray,
                            class_a: Optional[np.ndarray] = None,
                            class_b: Optional[np.ndarray] = None,
                            class_names: Optional[Dict[int, str]] = None,
                            min_area_pixels: int = 10,
                            pixel_size: Optional[float] = None) -> Dict:
    H, W = change_mask.shape

    polygons = mask_to_polygons(change_mask, min_area_pixels=min_area_pixels)

    features = []

    for idx, poly in enumerate(polygons):
        poly_coords = np.array(poly['coordinates'][0][:-1])

        x_coords = poly_coords[:, 0]
        y_coords = poly_coords[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        x_min = max(0, x_min)
        x_max = min(W - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(H - 1, y_max)

        mask_region = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        shifted_coords = poly_coords - np.array([x_min, y_min])
        cv2.fillPoly(mask_region, [shifted_coords.astype(np.int32)], 1)

        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[y_min:y_max+1, x_min:x_max+1] = mask_region.astype(bool)
        full_mask = full_mask & change_mask

        region_intensity = intensity_map[full_mask]
        mean_intensity = float(np.mean(region_intensity)) if len(region_intensity) > 0 else 0.0

        area_pixels = poly['area_pixels']
        area_m2 = area_pixels * (pixel_size ** 2) if pixel_size is not None else None

        transition_info = None
        if class_a is not None and class_b is not None:
            classes_a = class_a[full_mask]
            classes_b = class_b[full_mask]
            if len(classes_a) > 0 and len(classes_b) > 0:
                unique_a, counts_a = np.unique(classes_a, return_counts=True)
                unique_b, counts_b = np.unique(classes_b, return_counts=True)
                dominant_a = int(unique_a[np.argmax(counts_a)])
                dominant_b = int(unique_b[np.argmax(counts_b)])

                name_a = class_names.get(dominant_a, f'Class {dominant_a}') if class_names else f'Class {dominant_a}'
                name_b = class_names.get(dominant_b, f'Class {dominant_b}') if class_names else f'Class {dominant_b}'

                transition_info = {
                    'from_class_id': dominant_a,
                    'from_class_name': name_a,
                    'to_class_id': dominant_b,
                    'to_class_name': name_b,
                    'transition': f'{name_a} → {name_b}',
                }

        properties = {
            'id': idx + 1,
            'area_pixels': area_pixels,
            'mean_change_intensity': round(mean_intensity, 6),
            'max_change_intensity': round(float(np.max(region_intensity)), 6) if len(region_intensity) > 0 else 0.0,
            'min_change_intensity': round(float(np.min(region_intensity)), 6) if len(region_intensity) > 0 else 0.0,
            'std_change_intensity': round(float(np.std(region_intensity)), 6) if len(region_intensity) > 0 else 0.0,
        }

        if area_m2 is not None:
            properties['area_m2'] = round(area_m2, 4)
            properties['area_hectare'] = round(area_m2 / 10000.0, 6)

        if transition_info is not None:
            properties.update(transition_info)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': poly['coordinates']
            },
            'properties': properties
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': 'urn:ogc:def:crs:EPSG::4326'
            }
        },
        'metadata': {
            'image_dimensions': {'height': H, 'width': W},
            'total_change_regions': len(features),
            'total_change_pixels': int(np.sum(change_mask)),
            'total_pixels': int(change_mask.size),
            'change_ratio': round(float(np.sum(change_mask) / change_mask.size), 6),
        },
        'features': features
    }

    return geojson


def compute_kappa_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    flat1 = mask1.ravel().astype(np.int32)
    flat2 = mask2.ravel().astype(np.int32)
    return float(cohen_kappa_score(flat1, flat2))


def compute_overlap_heatmap(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask required")

    H, W = masks[0].shape
    for m in masks:
        if m.shape != (H, W):
            raise ValueError("All masks must have the same shape")

    overlap = np.zeros((H, W), dtype=np.int32)
    for m in masks:
        overlap += m.astype(np.int32)

    return overlap


def run_multi_algorithm_comparison(img_a: np.ndarray,
                                    img_b: np.ndarray,
                                    algorithms: List[str],
                                    params: Optional[Dict] = None,
                                    chunk_size: int = 500,
                                    progress_callback=None) -> Dict:
    if params is None:
        params = {}

    results = {}
    total_algos = len(algorithms)

    for idx, algo in enumerate(algorithms):
        if progress_callback:
            progress_callback(idx / total_algos, f"运行 {algo} 算法 ({idx+1}/{total_algos})...")

        if algo == 'SAD':
            threshold = params.get('SAD', {}).get('threshold', 0.1)
            mask, intensity, stats = sad_change_detection(
                img_a, img_b, threshold=threshold,
                chunk_size=chunk_size
            )
        elif algo == 'CVA':
            algo_params = params.get('CVA', {})
            mask, intensity, stats = cva_change_detection(
                img_a, img_b,
                threshold=algo_params.get('threshold'),
                threshold_method=algo_params.get('threshold_method', 'percentile'),
                percentile=algo_params.get('percentile', 95.0),
                chunk_size=chunk_size
            )
        elif algo == 'PCA':
            variance_ratio = params.get('PCA', {}).get('variance_ratio', 0.95)
            mask, intensity, stats = pca_change_detection(
                img_a, img_b, variance_ratio=variance_ratio,
                chunk_size=chunk_size
            )
        else:
            continue

        results[algo] = {
            'mask': mask,
            'intensity': intensity,
            'stats': stats
        }

    if progress_callback:
        progress_callback(0.8, "计算一致性指标...")

    comparison = {
        'algorithm_results': results,
        'area_ratios': {},
        'kappa_matrix': {},
    }

    for algo, res in results.items():
        comparison['area_ratios'][algo] = res['stats']['change_ratio']

    algo_list = list(results.keys())
    for i in range(len(algo_list)):
        for j in range(i + 1, len(algo_list)):
            a1, a2 = algo_list[i], algo_list[j]
            kappa = compute_kappa_coefficient(results[a1]['mask'], results[a2]['mask'])
            comparison['kappa_matrix'][(a1, a2)] = kappa

    if progress_callback:
        progress_callback(0.9, "生成重叠度热力图...")

    masks_list = [results[a]['mask'] for a in algo_list]
    comparison['overlap_heatmap'] = compute_overlap_heatmap(masks_list)

    if progress_callback:
        progress_callback(1.0, "多算法对比完成")

    return comparison


def compute_ndvi_timeseries(images: List[np.ndarray],
                            wavelengths: Optional[List[np.ndarray]] = None) -> List[float]:
    from .visualization import compute_ndvi as _compute_ndvi
    ndvi_values = []
    for i, img in enumerate(images):
        wl = wavelengths[i] if wavelengths and i < len(wavelengths) else None
        ndvi_map = _compute_ndvi(img, wl)
        ndvi_values.append(float(np.mean(ndvi_map)))
    return ndvi_values


def compute_ndvi_pixel_timeseries(images: List[np.ndarray],
                                   x: int, y: int,
                                   wavelengths: Optional[List[np.ndarray]] = None) -> List[float]:
    from .visualization import compute_ndvi as _compute_ndvi
    ndvi_values = []
    for i, img in enumerate(images):
        H, W = img.shape[:2]
        if 0 <= y < H and 0 <= x < W:
            pixel_data = img[y:y+1, x:x+1, :]
            wl = wavelengths[i] if wavelengths and i < len(wavelengths) else None
            ndvi_map = _compute_ndvi(pixel_data, wl)
            ndvi_values.append(float(ndvi_map[0, 0]))
        else:
            ndvi_values.append(0.0)
    return ndvi_values


def prepare_chord_diagram_data(transition_matrix: np.ndarray,
                                classes: List[int],
                                class_names: Dict[int, str]) -> Dict:
    n = len(classes)
    labels = [class_names.get(c, f'Class {c}') for c in classes]

    matrix = transition_matrix.astype(np.float64).tolist()

    total = float(np.sum(transition_matrix))

    details = []
    for i in range(n):
        for j in range(n):
            if i != j and transition_matrix[i, j] > 0:
                count = int(transition_matrix[i, j])
                details.append({
                    'source_idx': i,
                    'target_idx': j,
                    'source': labels[i],
                    'target': labels[j],
                    'pixels': count,
                    'percentage': round(count / total * 100, 4) if total > 0 else 0.0,
                })

    return {
        'labels': labels,
        'matrix': matrix,
        'details': details,
        'n_classes': n,
        'total_pixels': int(total),
    }
