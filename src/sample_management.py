import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass, field
from collections import Counter
from imblearn.over_sampling import SMOTE
from .utils import reshape_for_classifier


@dataclass
class ROIRegion:
    label: int
    label_name: str
    coordinates: List[Tuple[int, int]]
    shape_type: str
    pixels: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class TrainingSamples:
    features: np.ndarray
    labels: np.ndarray
    locations: Optional[np.ndarray] = None
    class_names: Dict[int, str] = field(default_factory=dict)
    rois: List[ROIRegion] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.labels)

    @property
    def class_counts(self) -> Dict[int, int]:
        return dict(Counter(self.labels))

    @property
    def class_distribution(self) -> Dict[int, float]:
        counts = self.class_counts
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    @property
    def is_balanced(self) -> bool:
        if len(self.classes) < 2:
            return True
        counts = list(self.class_counts.values())
        return min(counts) / max(counts) > 0.3


def extract_labeled_samples(data: np.ndarray, labels: np.ndarray,
                            ignore_label: int = 0) -> TrainingSamples:
    if data.shape[:2] != labels.shape[:2]:
        raise ValueError("Data and labels must have the same spatial dimensions")

    data_flat = reshape_for_classifier(data)
    labels_flat = labels.reshape(-1)

    valid_mask = labels_flat != ignore_label
    valid_indices = np.where(valid_mask)[0]

    features = data_flat[valid_indices]
    y = labels_flat[valid_indices]

    H, W = data.shape[:2]
    rows = valid_indices // W
    cols = valid_indices % W
    locations = np.column_stack([rows, cols])

    return TrainingSamples(
        features=features,
        labels=y,
        locations=locations
    )


def get_pixels_in_polygon(height: int, width: int,
                          polygon_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    from PIL import Image, ImageDraw

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    flat_points = [(x, y) for y, x in polygon_points]
    draw.polygon(flat_points, fill=1)

    mask = np.array(img)
    y_coords, x_coords = np.where(mask == 1)
    pixels = [(int(y), int(x)) for y, x in zip(y_coords, x_coords)]

    return pixels


def get_pixels_in_rectangle(height: int, width: int,
                            top_left: Tuple[int, int],
                            bottom_right: Tuple[int, int]) -> List[Tuple[int, int]]:
    y1, x1 = top_left
    y2, x2 = bottom_right

    y1, y2 = sorted([y1, y2])
    x1, x2 = sorted([x1, x2])

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))

    pixels = []
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pixels.append((y, x))

    return pixels


def add_roi_to_samples(data: np.ndarray, samples: TrainingSamples,
                       roi: ROIRegion) -> TrainingSamples:
    H, W = data.shape[:2]

    if not roi.pixels:
        if roi.shape_type == 'rectangle' and len(roi.coordinates) == 2:
            roi.pixels = get_pixels_in_rectangle(H, W, roi.coordinates[0], roi.coordinates[1])
        elif roi.shape_type == 'polygon' and len(roi.coordinates) >= 3:
            roi.pixels = get_pixels_in_polygon(H, W, roi.coordinates)

    if not roi.pixels:
        return samples

    pixels_array = np.array(roi.pixels)
    features = data[pixels_array[:, 0], pixels_array[:, 1]]
    labels = np.full(len(roi.pixels), roi.label, dtype=np.int32)

    if samples.features.size == 0:
        new_samples = TrainingSamples(
            features=features,
            labels=labels,
            locations=pixels_array,
            class_names={roi.label: roi.label_name}
        )
    else:
        new_samples = TrainingSamples(
            features=np.vstack([samples.features, features]),
            labels=np.concatenate([samples.labels, labels]),
            locations=np.vstack([samples.locations, pixels_array]) if samples.locations is not None else pixels_array,
            class_names={**samples.class_names, roi.label: roi.label_name}
        )

    new_samples.rois = samples.rois + [roi]

    return new_samples


def apply_smote(samples: TrainingSamples,
                random_state: int = 42,
                k_neighbors: int = 5) -> TrainingSamples:
    if samples.n_samples == 0:
        return samples

    counts = samples.class_counts
    min_count = min(counts.values())

    if min_count <= k_neighbors:
        k_neighbors = max(1, min_count - 1)

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    try:
        features_resampled, labels_resampled = smote.fit_resample(
            samples.features, samples.labels
        )

        new_samples = TrainingSamples(
            features=features_resampled,
            labels=labels_resampled,
            class_names=samples.class_names
        )

        return new_samples
    except Exception as e:
        print(f"SMOTE failed: {e}")
        return samples


def split_samples(samples: TrainingSamples,
                  test_size: float = 0.3,
                  stratify: bool = True,
                  random_state: int = 42) -> Tuple[TrainingSamples, TrainingSamples]:
    from .classification import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        samples.features, samples.labels,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    train_samples = TrainingSamples(
        features=X_train,
        labels=y_train,
        class_names=samples.class_names
    )

    test_samples = TrainingSamples(
        features=X_test,
        labels=y_test,
        class_names=samples.class_names
    )

    return train_samples, test_samples


def get_sample_stats(samples: TrainingSamples) -> Dict:
    stats = {
        'n_samples': samples.n_samples,
        'n_classes': len(samples.classes),
        'class_counts': samples.class_counts,
        'class_distribution': samples.class_distribution,
        'is_balanced': samples.is_balanced,
        'balance_ratio': min(samples.class_counts.values()) / max(samples.class_counts.values())
        if len(samples.classes) > 0 else 1.0,
        'class_names': samples.class_names
    }

    return stats


def create_empty_samples() -> TrainingSamples:
    return TrainingSamples(
        features=np.empty((0, 0), dtype=np.float32),
        labels=np.empty(0, dtype=np.int32)
    )
