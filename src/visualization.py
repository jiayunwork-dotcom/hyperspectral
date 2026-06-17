import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from matplotlib import colors
from .utils import normalize_image, stretch_contrast


DEFAULT_CLASS_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
]


def create_colormap(n_colors: int, custom_colors: Optional[List[str]] = None) -> colors.ListedColormap:
    if custom_colors is not None and len(custom_colors) >= n_colors:
        color_list = custom_colors[:n_colors]
    else:
        color_list = DEFAULT_CLASS_COLORS[:n_colors]
        if len(color_list) < n_colors:
            from matplotlib import cm
            cmap = cm.get_cmap('tab20', n_colors)
            color_list = [cmap(i) for i in range(n_colors)]

    return colors.ListedColormap(color_list)


def get_rgb_composite(data: np.ndarray,
                      r_band: int, g_band: int, b_band: int,
                      stretch: bool = True,
                      percentile: Tuple[int, int] = (2, 98)) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("Data must be 3D array (H, W, B)")

    n_bands = data.shape[2]
    if any(b >= n_bands for b in [r_band, g_band, b_band]):
        raise ValueError("Band indices out of range")

    r = data[:, :, r_band]
    g = data[:, :, g_band]
    b = data[:, :, b_band]

    rgb = np.stack([r, g, b], axis=-1)

    if stretch:
        rgb = stretch_contrast(rgb, percentile[0], percentile[1])
    else:
        rgb = normalize_image(rgb)

    return rgb


def get_true_color(data: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                   **kwargs) -> np.ndarray:
    if wavelengths is not None and len(wavelengths) >= 3:
        r_idx = np.argmin(np.abs(wavelengths - 650))
        g_idx = np.argmin(np.abs(wavelengths - 550))
        b_idx = np.argmin(np.abs(wavelengths - 450))
    else:
        n_bands = data.shape[2]
        r_idx = min(50, n_bands - 1)
        g_idx = min(30, n_bands - 1)
        b_idx = min(10, n_bands - 1)

    return get_rgb_composite(data, r_idx, g_idx, b_idx, **kwargs)


def get_false_color(data: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
    if wavelengths is not None and len(wavelengths) >= 3:
        nir_idx = np.argmin(np.abs(wavelengths - 850))
        r_idx = np.argmin(np.abs(wavelengths - 650))
        g_idx = np.argmin(np.abs(wavelengths - 550))
    else:
        n_bands = data.shape[2]
        nir_idx = min(70, n_bands - 1)
        r_idx = min(50, n_bands - 1)
        g_idx = min(30, n_bands - 1)

    return get_rgb_composite(data, nir_idx, r_idx, g_idx, **kwargs)


def compute_ndvi(data: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                 nir_band: Optional[int] = None, red_band: Optional[int] = None) -> np.ndarray:
    if nir_band is None or red_band is None:
        if wavelengths is not None and len(wavelengths) >= 2:
            nir_band = np.argmin(np.abs(wavelengths - 850))
            red_band = np.argmin(np.abs(wavelengths - 650))
        else:
            n_bands = data.shape[2]
            nir_band = min(70, n_bands - 1)
            red_band = min(50, n_bands - 1)

    nir = data[:, :, nir_band].astype(np.float32)
    red = data[:, :, red_band].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi


def colormap_ndvi(ndvi: np.ndarray) -> np.ndarray:
    from matplotlib import cm

    ndvi_norm = (ndvi + 1) / 2
    cmap = cm.get_cmap('RdYlGn')
    colored = cmap(ndvi_norm)[:, :, :3]

    return colored


def classification_to_rgb(classification: np.ndarray,
                          class_colors: Optional[Union[Dict[int, str], List[str]]] = None,
                          class_names: Optional[Dict[int, str]] = None) -> Tuple[np.ndarray, Dict]:
    classes = np.unique(classification)
    classes = np.sort(classes)
    n_classes = len(classes)

    if class_colors is None:
        cmap = create_colormap(n_classes)
        color_list = [cmap(i) for i in range(n_classes)]
    elif isinstance(class_colors, list):
        cmap = create_colormap(n_classes, class_colors)
        color_list = [cmap(i) for i in range(n_classes)]
    elif isinstance(class_colors, dict):
        color_list = []
        for cls in classes:
            hex_color = class_colors.get(cls, DEFAULT_CLASS_COLORS[0])
            rgb = colors.hex2color(hex_color)
            color_list.append(rgb)

    rgb = np.zeros((*classification.shape, 3), dtype=np.float32)
    legend = {}

    for i, cls in enumerate(classes):
        mask = classification == cls
        rgb[mask] = color_list[i][:3]

        name = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        legend[cls] = {
            'name': name,
            'color': colors.to_hex(color_list[i][:3])
        }

    return rgb, legend


def overlay_classification(background: np.ndarray, classification_rgb: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    if background.shape[:2] != classification_rgb.shape[:2]:
        raise ValueError("Background and classification must have same spatial dimensions")

    if background.ndim == 2:
        background = np.stack([background] * 3, axis=-1)

    background = normalize_image(background)
    classification_rgb = classification_rgb.astype(np.float32)

    overlay = (1 - alpha) * background + alpha * classification_rgb

    return overlay


def extract_single_class(classification: np.ndarray, target_class: int,
                         background: Optional[np.ndarray] = None) -> np.ndarray:
    mask = classification == target_class

    if background is not None:
        if background.ndim == 2:
            background = np.stack([background] * 3, axis=-1)
        background = normalize_image(background)

        result = background.copy()
        result[~mask] *= 0.2
        result[mask] = np.array([1, 0, 0])
    else:
        result = mask.astype(np.float32)

    return result


def create_band_composite(data: np.ndarray, bands: List[int],
                          **kwargs) -> np.ndarray:
    if len(bands) == 3:
        return get_rgb_composite(data, bands[0], bands[1], bands[2], **kwargs)
    elif len(bands) == 1:
        img = data[:, :, bands[0]]
        return normalize_image(img)
    else:
        raise ValueError("Band composite requires 1 or 3 bands")


def plot_spectrum(spectrum: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                  title: str = "Spectral Signature") -> Dict:
    if wavelengths is None:
        wavelengths = np.arange(len(spectrum))

    return {
        'wavelengths': wavelengths,
        'spectrum': spectrum,
        'title': title,
        'xlabel': 'Wavelength' if wavelengths is not None else 'Band Index',
        'ylabel': 'Reflectance'
    }


def plot_mnf_variance(explained_variance_ratio: np.ndarray) -> Dict:
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = len(explained_variance_ratio)

    return {
        'components': np.arange(1, n_components + 1),
        'individual': explained_variance_ratio,
        'cumulative': cumulative,
        'title': 'MNF Variance Explained'
    }


def plot_pca_variance(explained_variance_ratio: np.ndarray) -> Dict:
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = len(explained_variance_ratio)

    return {
        'components': np.arange(1, n_components + 1),
        'individual': explained_variance_ratio,
        'cumulative': cumulative,
        'title': 'PCA Variance Explained'
    }


def plot_confusion_matrix(cm: np.ndarray,
                          tick_labels: List[str],
                          normalize: bool = False,
                          title: str = "Confusion Matrix") -> Dict:
    return {
        'matrix': cm,
        'tick_labels': tick_labels,
        'normalized': normalize,
        'title': title
    }


def plot_class_distribution(class_counts: Dict[int, int],
                            class_names: Optional[Dict[int, str]] = None) -> Dict:
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    if class_names is None:
        labels = [f"Class {c}" for c in classes]
    else:
        labels = [class_names.get(c, f"Class {c}") for c in classes]

    return {
        'classes': classes,
        'counts': counts,
        'labels': labels,
        'title': 'Class Distribution'
    }


def generate_classification_legend(legend: Dict[int, Dict]) -> str:
    html_lines = ['<div style="display: flex; flex-wrap: wrap; gap: 10px;">']
    for cls, info in legend.items():
        html_lines.append(
            f'<div style="display: flex; align-items: center; gap: 5px;">'
            f'<div style="width: 20px; height: 20px; background-color: {info["color"]}; '
            f'border: 1px solid #ccc;"></div>'
            f'<span>{info["name"]}</span>'
            f'</div>'
        )
    html_lines.append('</div>')
    return ''.join(html_lines)


def create_error_spatial_map(labels: np.ndarray, predictions: np.ndarray,
                            background: np.ndarray = None,
                            alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have same shape")

    H, W = labels.shape
    error_mask = (labels != predictions) & (labels > 0)

    if background is None:
        display = np.zeros((H, W, 3), dtype=np.float32)
    else:
        display = normalize_image(background.copy()).astype(np.float32)
        if display.ndim == 2:
            display = np.stack([display] * 3, axis=-1)

    overlay = display.copy()
    overlay[error_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    display = (1 - alpha) * display + alpha * overlay

    return display, error_mask


def create_comparison_diff_map(result1: np.ndarray, result2: np.ndarray,
                                background: np.ndarray = None,
                                alpha: float = 0.5) -> Tuple[np.ndarray, Dict]:
    if result1.shape != result2.shape:
        raise ValueError("Results must have same shape")

    H, W = result1.shape
    agree_mask = result1 == result2
    disagree_mask = ~agree_mask

    if background is None:
        display = np.ones((H, W, 3), dtype=np.float32) * 0.5
    else:
        display = normalize_image(background.copy()).astype(np.float32)
        if display.ndim == 2:
            display = np.stack([display] * 3, axis=-1)

    display[agree_mask] = display[agree_mask] * 0.4 + np.array([0.5, 0.5, 0.5]) * 0.6

    diff_only1 = np.zeros((H, W, 3), dtype=np.float32)
    diff_only2 = np.zeros((H, W, 3), dtype=np.float32)

    result1_unique = result1[disagree_mask]
    result2_unique = result2[disagree_mask]

    all_classes = sorted(np.unique(np.concatenate([result1, result2])).tolist())
    cmap = create_colormap(len(all_classes))
    class_color_map = {}
    for i, cls in enumerate(all_classes):
        color = cmap(i)[:3]
        class_color_map[cls] = np.array(color, dtype=np.float32)

    result1_colored = np.zeros((H, W, 3), dtype=np.float32)
    result2_colored = np.zeros((H, W, 3), dtype=np.float32)
    for cls, color in class_color_map.items():
        m1 = (result1 == cls) & disagree_mask
        m2 = (result2 == cls) & disagree_mask
        result1_colored[m1] = color
        result2_colored[m2] = color

    display[disagree_mask] = (1 - alpha) * display[disagree_mask] + alpha * (
        result1_colored[disagree_mask] * 0.5 + result2_colored[disagree_mask] * 0.5
    )

    stats = {
        'total_pixels': H * W,
        'agree_pixels': int(np.sum(agree_mask)),
        'disagree_pixels': int(np.sum(disagree_mask)),
        'agreement_ratio': float(np.sum(agree_mask) / (H * W))
    }

    return display, stats
