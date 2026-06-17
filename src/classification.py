import numpy as np
from typing import Tuple, Optional, List, Dict, Union, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .utils import (
    reshape_for_classifier, reshape_back_to_image, chunk_generator,
    save_temp_array, load_temp_array, random_sample_indices
)


class HSIClassifier:
    def __init__(self, classifier_type: str, **kwargs):
        self.classifier_type = classifier_type
        self.params = kwargs
        self.model = None
        self.scaler = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            progress_callback=None) -> Dict:
        raise NotImplementedError

    def predict(self, X: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
        raise NotImplementedError


class SVMClassifier(HSIClassifier):
    def __init__(self, C: Optional[List[float]] = None,
                 gamma: Optional[List[float]] = None,
                 grid_search: bool = True,
                 cv: int = 3):
        super().__init__('svm', C=C, gamma=gamma, grid_search=grid_search, cv=cv)
        self.grid_search = grid_search
        self.C = C if C is not None else [0.1, 1, 10, 100]
        self.gamma = gamma if gamma is not None else ['scale', 0.001, 0.01, 0.1, 1]
        self.cv = cv

    def fit(self, X: np.ndarray, y: np.ndarray,
            progress_callback=None) -> Dict:
        if progress_callback:
            progress_callback(0.1, "Standardizing features...")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.classes_ = np.unique(y)

        if progress_callback:
            progress_callback(0.3, "Training SVM...")

        if self.grid_search:
            param_grid = {'C': self.C, 'gamma': self.gamma}
            base_svm = SVC(kernel='rbf', probability=True, random_state=42)

            if progress_callback:
                progress_callback(0.4, "Performing grid search...")

            grid_search = GridSearchCV(
                base_svm, param_grid, cv=self.cv,
                scoring='accuracy', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_

            if progress_callback:
                progress_callback(1.0, f"SVM trained. Best params: {grid_search.best_params_}")

            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        else:
            self.model = SVC(
                kernel='rbf', C=self.C[0] if isinstance(self.C, list) else self.C,
                gamma=self.gamma[0] if isinstance(self.gamma, list) else self.gamma,
                probability=True, random_state=42
            )
            self.model.fit(X_scaled, y)

            if progress_callback:
                progress_callback(1.0, "SVM trained")

            return {}

    def predict(self, X: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)

        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int32)

        if n_samples > 10000:
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                predictions[start:end] = self.model.predict(X_scaled[start:end])

                if progress_callback:
                    progress = end / n_samples
                    progress_callback(progress, f"Predicting samples {start}-{end}/{n_samples}")
        else:
            predictions = self.model.predict(X_scaled)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RandomForestClassifierHSI(HSIClassifier):
    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 max_features: Union[str, float] = 'sqrt',
                 min_samples_split: int = 2):
        super().__init__('random_forest', n_estimators=n_estimators,
                         max_depth=max_depth, max_features=max_features)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray,
            progress_callback=None) -> Dict:
        if progress_callback:
            progress_callback(0.2, "Training Random Forest...")

        self.classes_ = np.unique(y)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X, y)

        importances = self.model.feature_importances_

        if progress_callback:
            progress_callback(1.0, "Random Forest trained")

        return {
            'feature_importances': importances,
            'n_estimators': self.n_estimators
        }

    def predict(self, X: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int32)

        if n_samples > 10000:
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                predictions[start:end] = self.model.predict(X[start:end])

                if progress_callback:
                    progress = end / n_samples
                    progress_callback(progress, f"Predicting samples {start}-{end}/{n_samples}")
        else:
            predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class OneDCNN(nn.Module):
    def __init__(self, n_bands: int, n_classes: int):
        super(OneDCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class OneDCNNClassifier(HSIClassifier):
    def __init__(self, n_epochs: int = 50,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        super().__init__('1d_cnn', n_epochs=n_epochs,
                         batch_size=batch_size, learning_rate=learning_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X: np.ndarray, y: np.ndarray,
            progress_callback=None) -> Dict:
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_bands = X.shape[1]

        label_map = {cls: i for i, cls in enumerate(self.classes_)}
        y_mapped = np.array([label_map[cls] for cls in y])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_mapped).to(self.device)

        n_samples = len(X)
        if n_samples > 10000:
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = OneDCNN(n_bands, n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        train_accs = []

        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

            if progress_callback:
                progress = (epoch + 1) / self.n_epochs
                progress_callback(progress,
                                f"Epoch {epoch+1}/{self.n_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

        return {
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'n_classes': n_classes,
            'label_map': label_map
        }

    def predict(self, X: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int32)

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = torch.FloatTensor(X_scaled[start:end]).to(self.device)
                outputs = self.model(batch)
                _, predicted = torch.max(outputs.data, 1)
                predictions[start:end] = predicted.cpu().numpy()

                if progress_callback:
                    progress = end / n_samples
                    progress_callback(progress, f"Predicting samples {start}-{end}/{n_samples}")

        inv_label_map = {i: cls for cls, i in self.params.get('label_map', {}).items()}
        if inv_label_map:
            predictions = np.array([inv_label_map[p] for p in predictions])

        return predictions


class ThreeDCNN(nn.Module):
    def __init__(self, window_size: int, n_bands: int, n_classes: int):
        super(ThreeDCNN, self).__init__()
        self.window_size = window_size
        self.n_bands = n_bands

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThreeDCNNClassifier(HSIClassifier):
    def __init__(self, window_size: int = 7,
                 n_epochs: int = 50,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        super().__init__('3d_cnn', window_size=window_size, n_epochs=n_epochs,
                         batch_size=batch_size, learning_rate=learning_rate)
        self.window_size = window_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.full_data_shape = None

    def _extract_patches(self, data: np.ndarray, locations: np.ndarray) -> np.ndarray:
        H, W, B = data.shape
        half_w = self.window_size // 2
        n_samples = len(locations)

        patches = np.zeros((n_samples, self.window_size, self.window_size, B), dtype=np.float32)

        for i, (r, c) in enumerate(locations):
            r_start = max(0, r - half_w)
            r_end = min(H, r + half_w + 1)
            c_start = max(0, c - half_w)
            c_end = min(W, c + half_w + 1)

            pr_start = half_w - (r - r_start)
            pr_end = pr_start + (r_end - r_start)
            pc_start = half_w - (c - c_start)
            pc_end = pc_start + (c_end - c_start)

            patches[i, pr_start:pr_end, pc_start:pc_end, :] = data[r_start:r_end, c_start:c_end, :]

        return patches

    def fit(self, X: np.ndarray, y: np.ndarray,
            data_image: Optional[np.ndarray] = None,
            progress_callback=None) -> Dict:
        if data_image is None:
            raise ValueError("data_image is required for 3D CNN")

        self.full_data_shape = data_image.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        H, W, B = data_image.shape

        label_map = {cls: i for i, cls in enumerate(self.classes_)}
        y_mapped = np.array([label_map[cls] for cls in y])

        self.scaler = StandardScaler()
        data_flat = reshape_for_classifier(data_image)
        self.scaler.fit(data_flat)
        data_scaled = self.scaler.transform(data_flat).reshape(H, W, B)

        n_train = len(y)
        rows = np.random.randint(0, H, n_train)
        cols = np.random.randint(0, W, n_train)
        locations = np.column_stack([rows, cols])

        patches = self._extract_patches(data_scaled, locations)

        X_tensor = torch.FloatTensor(patches).to(self.device)
        y_tensor = torch.LongTensor(y_mapped).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = ThreeDCNN(self.window_size, B, n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        train_accs = []

        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

            if progress_callback:
                progress = (epoch + 1) / self.n_epochs
                progress_callback(progress,
                                f"Epoch {epoch+1}/{self.n_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

        return {
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'n_classes': n_classes,
            'label_map': label_map,
            'window_size': self.window_size
        }

    def predict(self, X_image: np.ndarray,
                chunk_size: int = 1000,
                overlap: int = 32,
                progress_callback=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        H, W, B = X_image.shape

        data_flat = reshape_for_classifier(X_image)
        data_scaled = self.scaler.transform(data_flat).reshape(H, W, B)

        predictions = np.zeros((H, W), dtype=np.int32)
        count_map = np.zeros((H, W), dtype=np.int32)

        patch_size = self.window_size
        stride = chunk_size - overlap
        half_w = patch_size // 2

        total_blocks = ((H + stride - 1) // stride) * ((W + stride - 1) // stride)
        block_idx = 0

        with torch.no_grad():
            for r_start in range(0, H, stride):
                r_end = min(r_start + chunk_size, H)
                r_pad_start = max(0, r_start - half_w)
                r_pad_end = min(H, r_end + half_w)

                for c_start in range(0, W, stride):
                    c_end = min(c_start + chunk_size, W)
                    c_pad_start = max(0, c_start - half_w)
                    c_pad_end = min(W, c_end + half_w)

                    block = data_scaled[r_pad_start:r_pad_end, c_pad_start:c_pad_end]
                    bh, bw, _ = block.shape

                    n_pixels = (r_end - r_start) * (c_end - c_start)
                    rows = np.arange(r_start, r_end)
                    cols = np.arange(c_start, c_end)
                    rr, cc = np.meshgrid(rows, cols, indexing='ij')
                    locations = np.column_stack([rr.ravel(), cc.ravel()])

                    local_locations = locations - np.array([[r_pad_start, c_pad_start]])

                    patches = self._extract_patches(block, local_locations)

                    batch_predictions = np.zeros(len(patches), dtype=np.int32)
                    for b_start in range(0, len(patches), self.batch_size):
                        b_end = min(b_start + self.batch_size, len(patches))
                        batch = torch.FloatTensor(patches[b_start:b_end]).to(self.device)
                        outputs = self.model(batch)
                        _, predicted = torch.max(outputs.data, 1)
                        batch_predictions[b_start:b_end] = predicted.cpu().numpy()

                    pred_block = batch_predictions.reshape(r_end - r_start, c_end - c_start)
                    predictions[r_start:r_end, c_start:c_end] += pred_block
                    count_map[r_start:r_end, c_start:c_end] += 1

                    block_idx += 1
                    if progress_callback:
                        progress = block_idx / total_blocks
                        progress_callback(progress,
                                        f"Predicting block {block_idx}/{total_blocks}")

        predictions = predictions / np.maximum(count_map, 1)
        predictions = predictions.astype(np.int32)

        inv_label_map = {i: cls for cls, i in self.params.get('label_map', {}).items()}
        if inv_label_map:
            predictions = np.vectorize(lambda x: inv_label_map.get(x, x))(predictions)

        return predictions


class SemiSupervisedClassifier(HSIClassifier):
    def __init__(self, gamma: float = 20,
                 max_iter: int = 1000,
                 n_neighbors: int = 7):
        super().__init__('semi_supervised', gamma=gamma,
                         max_iter=max_iter, n_neighbors=n_neighbors)
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_unlabeled: Optional[np.ndarray] = None,
            progress_callback=None) -> Dict:
        if progress_callback:
            progress_callback(0.2, "Preparing data for label propagation...")

        self.scaler = StandardScaler()

        if X_unlabeled is not None:
            X_all = np.vstack([X, X_unlabeled])
            y_all = np.concatenate([y, np.full(len(X_unlabeled), -1)])
        else:
            X_all = X
            y_all = y

        X_all_scaled = self.scaler.fit_transform(X_all)
        self.classes_ = np.unique(y[y != -1])

        if progress_callback:
            progress_callback(0.5, "Training Label Propagation...")

        self.model = LabelPropagation(
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_neighbors=self.n_neighbors
        )
        self.model.fit(X_all_scaled, y_all)

        if progress_callback:
            progress_callback(1.0, "Label Propagation complete")

        return {
            'n_labeled': np.sum(y_all != -1),
            'n_unlabeled': np.sum(y_all == -1),
            'n_iterations': self.model.n_iter_
        }

    def predict(self, X: np.ndarray,
                chunk_size: int = 1000,
                progress_callback=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions


def create_classifier(classifier_type: str, **kwargs) -> HSIClassifier:
    classifiers = {
        'svm': SVMClassifier,
        'random_forest': RandomForestClassifierHSI,
        '1d_cnn': OneDCNNClassifier,
        '3d_cnn': ThreeDCNNClassifier,
        'semi_supervised': SemiSupervisedClassifier,
    }

    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return classifiers[classifier_type](**kwargs)


def classify_image(classifier: HSIClassifier,
                   features: np.ndarray,
                   chunk_size: int = 1000,
                   progress_callback=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    H, W = features.shape[:2]
    features_flat = reshape_for_classifier(features)

    if isinstance(classifier, ThreeDCNNClassifier):
        predictions = classifier.predict(
            features, chunk_size=chunk_size,
            progress_callback=progress_callback
        )
    else:
        predictions = classifier.predict(
            features_flat, chunk_size=chunk_size,
            progress_callback=progress_callback
        )

    if predictions.ndim == 1:
        predictions = predictions.reshape(H, W)

    return predictions, None


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.3,
                     stratify: bool = True,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stratify:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(sss.split(X, y))
    else:
        rng = np.random.RandomState(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_hyperparameter_experiment(classifier_key: str,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  param_grid: Dict[str, List],
                                  progress_callback=None,
                                  class_names: Dict[int, str] = None) -> List[Dict]:
    from itertools import product
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from sklearn.preprocessing import StandardScaler

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    all_combinations = list(product(*param_values))

    results = []
    total = len(all_combinations)

    for idx, combo in enumerate(all_combinations):
        if progress_callback:
            progress = (idx + 0.3) / total
            progress_callback(progress, f"测试组合 {idx+1}/{total}")

        params = dict(zip(param_names, combo))
        try:
            if classifier_key == 'svm':
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                C = params['C']
                gamma = params['gamma'] if params['gamma'] != 'scale' else 'scale'
                model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_val_s)

            elif classifier_key == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    max_features=params.get('max_features', 'sqrt'),
                    n_jobs=-1, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

            else:
                continue

            oa = accuracy_score(y_val, y_pred)
            kappa = cohen_kappa_score(y_val, y_pred)

            results.append({
                'params': params,
                'oa': float(oa),
                'kappa': float(kappa),
                'index': idx
            })

        except Exception as e:
            results.append({
                'params': params,
                'oa': 0.0,
                'kappa': 0.0,
                'index': idx,
                'error': str(e)
            })

    if progress_callback:
        progress_callback(1.0, f"完成 {total} 组实验")

    return results


def generate_param_grid_linear(start: float, end: float, n_points: int) -> List[float]:
    return np.linspace(start, end, n_points).tolist()


def generate_param_grid_log(start: float, end: float, n_points: int) -> List[float]:
    return np.logspace(np.log10(start), np.log10(end), n_points).tolist()
