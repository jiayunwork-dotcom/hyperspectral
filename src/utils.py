import numpy as np
import tempfile
import os
import re
from typing import Tuple, Optional, Union, Dict, List


def get_temp_file(suffix: str = '') -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def save_temp_array(arr: np.ndarray) -> str:
    path = get_temp_file('.npy')
    np.save(path, arr)
    return path


def load_temp_array(path: str, mmap: bool = True) -> np.ndarray:
    if mmap:
        return np.load(path, mmap_mode='r')
    return np.load(path)


def normalize_image(img: np.ndarray, percentile: Tuple[int, int] = (2, 98)) -> np.ndarray:
    img = img.astype(np.float32)
    low = np.percentile(img, percentile[0])
    high = np.percentile(img, percentile[1])
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    return img


def stretch_contrast(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    return normalize_image(img, (pmin, pmax))


def reshape_for_classifier(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        n_samples = X.shape[0] * X.shape[1]
        n_features = X.shape[2]
        return X.reshape(n_samples, n_features)
    return X


def reshape_back_to_image(X: np.ndarray, height: int, width: int) -> np.ndarray:
    n_classes = X.shape[1] if X.ndim > 1 else 1
    if n_classes > 1:
        return X.reshape(height, width, n_classes)
    return X.reshape(height, width)


def get_memory_usage_mb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 2)


def chunk_generator(data: np.ndarray, chunk_size: int = 1000, axis: int = 0):
    n = data.shape[axis]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if axis == 0:
            yield data[start:end], start, end
        elif axis == 1:
            yield data[:, start:end], start, end
        else:
            yield data[..., start:end], start, end


def random_sample_indices(n_total: int, n_sample: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if n_total <= n_sample:
        return np.arange(n_total)
    return rng.choice(n_total, n_sample, replace=False)


def interleave_to_numpy(data: np.ndarray, interleave: str, shape: Tuple[int, int, int]) -> np.ndarray:
    n_lines, n_samples, n_bands = shape
    interleave = interleave.lower()
    if interleave == 'bsq':
        return data.reshape(n_bands, n_lines, n_samples).transpose(1, 2, 0)
    elif interleave == 'bil':
        return data.reshape(n_lines, n_bands, n_samples).transpose(0, 2, 1)
    elif interleave == 'bip':
        return data.reshape(n_lines, n_samples, n_bands)
    else:
        raise ValueError(f"Unknown interleave format: {interleave}")


def numpy_to_interleave(data: np.ndarray, interleave: str) -> np.ndarray:
    n_lines, n_samples, n_bands = data.shape
    interleave = interleave.lower()
    if interleave == 'bsq':
        return data.transpose(2, 0, 1).reshape(-1)
    elif interleave == 'bil':
        return data.transpose(0, 2, 1).reshape(-1)
    elif interleave == 'bip':
        return data.reshape(-1)
    else:
        raise ValueError(f"Unknown interleave format: {interleave}")


def get_dtype_from_envi(data_type: int) -> np.dtype:
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    if data_type not in dtype_map:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")
    return dtype_map[data_type]


def get_envi_dtype(dtype: np.dtype) -> int:
    if dtype == np.uint8:
        return 1
    elif dtype == np.int16:
        return 2
    elif dtype == np.int32:
        return 3
    elif dtype == np.float32:
        return 4
    elif dtype == np.float64:
        return 5
    elif dtype == np.uint16:
        return 12
    elif dtype == np.uint32:
        return 13
    elif dtype == np.int64:
        return 14
    elif dtype == np.uint64:
        return 15
    else:
        raise ValueError(f"Unsupported dtype for ENVI: {dtype}")


def parse_map_info(map_info_str: str) -> Optional[Dict]:
    if not map_info_str or not map_info_str.strip():
        return None

    map_info_str = map_info_str.strip()
    if map_info_str.startswith('{'):
        map_info_str = map_info_str[1:]
    if map_info_str.endswith('}'):
        map_info_str = map_info_str[:-1]

    parts = [p.strip() for p in map_info_str.split(',')]

    if len(parts) < 7:
        return None

    try:
        proj_name = parts[0]
        ref_pixel_x = float(parts[1])
        ref_pixel_y = float(parts[2])
        ref_easting = float(parts[3])
        ref_northing = float(parts[4])
        pixel_size_x = float(parts[5])
        pixel_size_y = float(parts[6])

        projection = ''
        datum = ''
        units = 'Meters'

        if len(parts) > 7:
            for i in range(7, len(parts)):
                p = parts[i].strip()
                if p.lower() in ['meters', 'feet', 'us feet']:
                    units = p
                elif 'WGS' in p.upper() or 'NAD' in p.upper() or 'DATUM' in p.upper():
                    datum = p

        return {
            'projection': proj_name,
            'reference_pixel': (ref_pixel_x, ref_pixel_y),
            'reference_coords': (ref_easting, ref_northing),
            'pixel_size': (pixel_size_x, pixel_size_y),
            'units': units,
            'datum': datum,
            'rotation': 0.0,
        }
    except (ValueError, IndexError):
        return None


def pixel_to_geo(x: float, y: float, map_info: Dict) -> Tuple[float, float]:
    ref_x, ref_y = map_info['reference_pixel']
    ref_e, ref_n = map_info['reference_coords']
    ps_x, ps_y = map_info['pixel_size']

    geo_x = ref_e + (x - ref_x + 1) * ps_x
    geo_y = ref_n - (y - ref_y + 1) * ps_y

    return geo_x, geo_y


def pixels_to_geo_coords(coords: List[List[float]], map_info: Dict) -> List[List[float]]:
    geo_coords = []
    for x, y in coords:
        gx, gy = pixel_to_geo(x, y, map_info)
        geo_coords.append([gx, gy])
    return geo_coords
