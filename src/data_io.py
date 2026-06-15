import numpy as np
import os
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, field
from .utils import get_dtype_from_envi, get_temp_file


@dataclass
class ENVIHeader:
    lines: int = 0
    samples: int = 0
    bands: int = 0
    data_type: int = 4
    interleave: str = 'bip'
    byte_order: int = 0
    header_offset: int = 0
    wavelengths: List[float] = field(default_factory=list)
    fwhm: List[float] = field(default_factory=list)
    band_names: List[str] = field(default_factory=list)
    map_info: str = ''
    coordinate_system_string: str = ''
    data_ignore_value: Optional[float] = None
    additional_fields: Dict = field(default_factory=dict)


def parse_envi_header(hdr_path: str) -> ENVIHeader:
    header = ENVIHeader()

    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    lines = content.splitlines()

    list_fields = ['wavelength', 'fwhm', 'band names', 'wavelengths']
    current_list_field = None
    current_list_values = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.lower().startswith('envi'):
            i += 1
            continue

        if '=' in line and not current_list_field:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()

            if value.startswith('{'):
                if not value.endswith('}'):
                    current_list_field = key
                    current_list_values = [value[1:].strip()]
                else:
                    values_str = value[1:-1].strip()
                    _parse_list_field(header, key, values_str)
            else:
                _parse_scalar_field(header, key, value)
        elif current_list_field:
            if '}' in line:
                end_idx = line.index('}')
                current_list_values.append(line[:end_idx].strip())
                values_str = ','.join([v for v in current_list_values if v])
                _parse_list_field(header, current_list_field, values_str)
                current_list_field = None
            else:
                current_list_values.append(line.strip())

        i += 1

    return header


def _parse_scalar_field(header: ENVIHeader, key: str, value: str):
    key_map = {
        'lines': 'lines',
        'samples': 'samples',
        'bands': 'bands',
        'data type': 'data_type',
        'interleave': 'interleave',
        'byte order': 'byte_order',
        'header offset': 'header_offset',
        'data ignore value': 'data_ignore_value',
        'map info': 'map_info',
        'coordinate system string': 'coordinate_system_string',
    }

    if key in key_map:
        attr = key_map[key]
        if attr in ['lines', 'samples', 'bands', 'data_type', 'byte_order', 'header_offset']:
            setattr(header, attr, int(value))
        elif attr == 'data_ignore_value':
            try:
                setattr(header, attr, float(value))
            except ValueError:
                pass
        else:
            setattr(header, attr, value)
    else:
        header.additional_fields[key] = value


def _parse_list_field(header: ENVIHeader, key: str, values_str: str):
    values = [v.strip() for v in values_str.split(',') if v.strip()]

    if key in ['wavelength', 'wavelengths']:
        try:
            header.wavelengths = [float(v) for v in values]
        except ValueError:
            header.wavelengths = []
    elif key == 'fwhm':
        try:
            header.fwhm = [float(v) for v in values]
        except ValueError:
            header.fwhm = []
    elif key == 'band names':
        header.band_names = values
    else:
        header.additional_fields[key] = values


def load_envi_data(hdr_path: str, dat_path: Optional[str] = None,
                   mmap: bool = True) -> Tuple[np.ndarray, ENVIHeader]:
    header = parse_envi_header(hdr_path)

    if dat_path is None:
        base, ext = os.path.splitext(hdr_path)
        dat_path = base

    dtype = get_dtype_from_envi(header.data_type)
    dtype_size = np.dtype(dtype).itemsize

    shape = (header.lines, header.samples, header.bands)
    total_elements = header.lines * header.samples * header.bands
    expected_size = total_elements * dtype_size

    if mmap:
        data = np.memmap(
            dat_path,
            dtype=dtype,
            mode='r',
            offset=header.header_offset,
            shape=(total_elements,)
        )
    else:
        with open(dat_path, 'rb') as f:
            f.seek(header.header_offset)
            data = np.fromfile(f, dtype=dtype, count=total_elements)

    if header.byte_order != 0:
        data = data.byteswap()

    from .utils import interleave_to_numpy
    data = interleave_to_numpy(data, header.interleave, shape)

    if mmap:
        tmp_path = get_temp_file('.npy')
        np.save(tmp_path, data)
        data = np.load(tmp_path, mmap_mode='r')

    return data, header


def load_envi_labels(label_path: str, hdr_path: Optional[str] = None) -> np.ndarray:
    if hdr_path and os.path.exists(hdr_path):
        header = parse_envi_header(hdr_path)
        labels, _ = load_envi_data(hdr_path, label_path, mmap=True)
        labels = labels.astype(np.int32)
        if labels.ndim == 3 and labels.shape[2] == 1:
            labels = labels[:, :, 0]
    else:
        labels = np.load(label_path) if label_path.endswith('.npy') else None
        if labels is None:
            try:
                labels = np.loadtxt(label_path, dtype=np.int32)
            except:
                from PIL import Image
                labels = np.array(Image.open(label_path), dtype=np.int32)

    return labels


def save_envi_data(data: np.ndarray, hdr_path: str, dat_path: Optional[str] = None,
                   wavelengths: Optional[List[float]] = None,
                   interleave: str = 'bip',
                   header_offset: int = 0):
    if dat_path is None:
        base, _ = os.path.splitext(hdr_path)
        dat_path = base

    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    n_lines, n_samples, n_bands = data.shape

    from .utils import numpy_to_interleave, get_envi_dtype
    flat_data = numpy_to_interleave(data, interleave)

    with open(dat_path, 'wb') as f:
        if header_offset > 0:
            f.write(b'\x00' * header_offset)
        flat_data.tofile(f)

    envi_dtype = get_envi_dtype(flat_data.dtype)

    hdr_content = [
        'ENVI',
        f'lignes = {n_lines}',
        f'echantillons = {n_samples}',
        f'bandes = {n_bands}',
        f'type de donnees = {envi_dtype}',
        f'entrelacement = {interleave.lower()}',
        f'ordre des octets = 0',
        f'decalage d\'entete = {header_offset}',
    ]

    if wavelengths and len(wavelengths) == n_bands:
        wl_str = ', '.join([f'{w:.2f}' for w in wavelengths])
        hdr_content.append(f'longueurs d\'onde = {{ {wl_str} }}')

    hdr_content.append('')

    with open(hdr_path, 'w') as f:
        f.write('\n'.join(hdr_content))


def get_image_info(data: np.ndarray, header: ENVIHeader) -> Dict:
    info = {
        'dimensions': f"{header.lines} x {header.samples}",
        'num_bands': header.bands,
        'data_type': str(data.dtype),
        'interleave': header.interleave,
        'wavelength_range': (min(header.wavelengths), max(header.wavelengths)) if header.wavelengths else None,
        'memory_mb': data.nbytes / (1024 ** 2),
        'map_info': header.map_info,
        'coordinate_system': header.coordinate_system_string,
    }
    return info
