import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: Optional[Dict[int, str]] = None) -> Dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    labels = sorted(classes.tolist())

    oa = accuracy_score(y_true, y_pred)

    precision_per_class = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    aa = np.mean(recall_per_class)

    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    per_class_metrics = {}
    for i, cls in enumerate(labels):
        per_class_metrics[cls] = {
            'name': class_names.get(cls, f"Class {cls}"),
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i]
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'overall_accuracy': oa,
        'average_accuracy': aa,
        'kappa': kappa,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'labels': labels,
        'n_classes': n_classes
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = False,
                             class_names: Optional[Dict[int, str]] = None) -> Dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    labels = sorted(classes.tolist())

    if class_names is None:
        class_names = {c: f"Class {c}" for c in labels}

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        matrix = cm_norm
    else:
        matrix = cm

    tick_labels = [class_names.get(c, f"Class {c}") for c in labels]

    return {
        'matrix': matrix,
        'raw_matrix': cm,
        'labels': labels,
        'tick_labels': tick_labels,
        'normalized': normalize
    }


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: Optional[Dict[int, str]] = None) -> str:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    labels = sorted(classes.tolist())

    if class_names is None:
        target_names = [f"Class {c}" for c in labels]
    else:
        target_names = [class_names.get(c, f"Class {c}") for c in labels]

    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )


def evaluate_classifier(classifier, X_test: np.ndarray, y_test: np.ndarray,
                        class_names: Optional[Dict[int, str]] = None,
                        progress_callback=None) -> Dict:
    if progress_callback:
        progress_callback(0.2, "Predicting on test set...")

    y_pred = classifier.predict(X_test)

    if progress_callback:
        progress_callback(0.6, "Computing metrics...")

    metrics = compute_metrics(y_test, y_pred, class_names)

    if progress_callback:
        progress_callback(1.0, "Evaluation complete")

    return {
        'metrics': metrics,
        'y_true': y_test,
        'y_pred': y_pred
    }


def compare_classifiers(results: List[Dict]) -> Dict:
    comparison = {
        'classifiers': [],
        'overall_accuracy': [],
        'average_accuracy': [],
        'kappa': []
    }

    for result in results:
        metrics = result['metrics']
        comparison['classifiers'].append(result.get('name', 'Unknown'))
        comparison['overall_accuracy'].append(metrics['overall_accuracy'])
        comparison['average_accuracy'].append(metrics['average_accuracy'])
        comparison['kappa'].append(metrics['kappa'])

    return comparison


def print_metrics_summary(metrics: Dict):
    print("=" * 60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy (OA):  {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"Average Accuracy (AA):  {metrics['average_accuracy']:.4f} ({metrics['average_accuracy']*100:.2f}%)")
    print(f"Kappa Coefficient:      {metrics['kappa']:.4f}")
    print(f"Number of Classes:      {metrics['n_classes']}")
    print("-" * 60)
    print("Per-Class Metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    for cls, m in metrics['per_class_metrics'].items():
        print(f"{m['name']:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("=" * 60)


def format_metrics_for_display(metrics: Dict) -> Dict:
    formatted = {
        'Overall Accuracy (OA)': f"{metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)",
        'Average Accuracy (AA)': f"{metrics['average_accuracy']:.4f} ({metrics['average_accuracy']*100:.2f}%)",
        'Kappa Coefficient': f"{metrics['kappa']:.4f}",
        'Number of Classes': str(metrics['n_classes'])
    }

    per_class = []
    for cls, m in metrics['per_class_metrics'].items():
        per_class.append({
            'Class': m['name'],
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1']:.4f}"
        })

    return {
        'summary': formatted,
        'per_class': per_class
    }
