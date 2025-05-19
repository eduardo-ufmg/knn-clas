import os
import time
import json
import numpy as np

from enum import IntEnum
from typing import Tuple, Self

from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import wilcoxon, ttest_rel # type: ignore

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted # type: ignore
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score # type: ignore
)

np.random.seed(0)

class Adjacency(IntEnum):
    NOT_ADJACENT = 0
    GABRIEL_EDGE = 1
    SUPPORT_EDGE = 2

def vectorized_kernel(X: np.ndarray, Y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian kernel computation."""
    n_features = X.shape[1]
    inv_cov = np.linalg.pinv(cov)
    X_diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    exponent = -0.5 * np.einsum('...i,ij,...j->...', X_diff, inv_cov, X_diff)
    det = np.linalg.det(cov + 1e-6 * np.eye(n_features))
    norm_factor = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
    return norm_factor * np.exp(exponent)

def gabriel_graph(dist_matrix: np.ndarray) -> np.ndarray:
    """Construct Gabriel graph from pairwise squared distance matrix."""
    n = dist_matrix.shape[0]
    adjacency = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i+1, n):
            d_ij = dist_matrix[i, j]
            other_dists = dist_matrix[i] + dist_matrix[j]
            if np.all(other_dists >= d_ij):
                adjacency[i, j] = Adjacency.GABRIEL_EDGE
                adjacency[j, i] = Adjacency.GABRIEL_EDGE
    return adjacency

def support_graph(gabriel_adj: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create support graph from Gabriel graph and labels."""
    support_adj = np.zeros_like(gabriel_adj)
    diff_labels = y[:, None] != y[None, :]
    support_edges = (gabriel_adj > 0) & diff_labels
    support_adj[support_edges] = Adjacency.SUPPORT_EDGE
    support_adj[(gabriel_adj > 0) & ~diff_labels] = Adjacency.GABRIEL_EDGE
    return support_adj

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.cov_ = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.atleast_2d(X)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_)
        scores = np.sum(kernels * self.y_train_[nearest], axis=1)
        
        return np.where(scores.mean(1) > 0, 1, -1)

    def likelihood_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_)
        q0 = np.sum(kernels * (self.y_train_[nearest] == -1), axis=1)
        q1 = np.sum(kernels * (self.y_train_[nearest] == 1), axis=1)

        q_total = q0 + q1
        q_sum = np.sum(q_total)
        q0_norm = q0 / q_sum
        q1_norm = q1 / q_sum
        
        return q0_norm, q1_norm

class KNN_CLAS(KNN):
    def __init__(self, k: int = 3):
        super().__init__(k=k)

    def _get_experts(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist_matrix = squareform(pdist(X, metric='sqeuclidean'))
        gabriel_adj = gabriel_graph(dist_matrix)
        support_adj = support_graph(gabriel_adj, y)
        expert_mask = np.any(support_adj == Adjacency.SUPPORT_EDGE, axis=1)
        
        if not np.any(expert_mask):
            print("No support edges found; using all points as experts.")
            return X, y
        
        return X[expert_mask], y[expert_mask]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        super().fit(X, y)
        self.expert_X_, self.expert_y_ = self._get_experts(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_)
        scores = np.sum(kernels * self.expert_y_[nearest], axis=1)
        
        return np.where(scores.mean(1) > 0, 1, -1)
    
    def likelihood_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_)
        q0 = np.sum(kernels * (self.expert_y_[nearest] == -1), axis=1)
        q1 = np.sum(kernels * (self.expert_y_[nearest] == 1), axis=1)

        q_total = q0 + q1
        q_sum = np.sum(q_total)
        q0_norm = q0 / q_sum
        q1_norm = q1 / q_sum
        
        return q0_norm, q1_norm

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.98):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr = np.where(np.abs(corr_matrix) > self.threshold)
        self.to_drop_ = set()
        for i, j in zip(*high_corr):
            if upper[i, j]:  # Avoid duplicates
                self.to_drop_.add(j)  # Drop the later feature
        return self

    def transform(self, X):
        return X[:, [i for i in range(X.shape[1]) if i not in self.to_drop_]]

def load_datasets(data_dir: str = './sets') -> dict:
    datasets = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('.npz'):
            data = np.load(os.path.join(data_dir, fname))
            name = fname.split('.')[0]
            X = data['X']
            y = data['y'].squeeze()
            datasets[name] = {'X': X, 'y': y}
            print(f"Loaded {name}: {X.shape}")
    return datasets

def run_statistical_validation(datasets: dict, output_dir: str = 'output') -> None:
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    preprocessor = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=1e-3)),
        ('correlation_filter', CorrelationFilter(threshold=0.9)),
        ('scaler', StandardScaler())
    ])

    for dname, data in datasets.items():
        print(f"\n=== Processing {dname} ===")
        X = data['X']
        y = data['y']
        
        try:
            cv_metrics = {
                'KNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []},
                'KNN_CLAS': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []}
            }
            
            for train_idx, test_idx in KFold(n_splits=30).split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                for model in [
                        Pipeline([('preprocessor', preprocessor), ('classifier', KNN(k=5))]),
                        Pipeline([('preprocessor', preprocessor), ('classifier', KNN_CLAS(k=5))])
                    ]:
                    mname = model.named_steps['classifier'].__class__.__name__
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    cv_metrics[mname]['accuracy'].append(accuracy_score(y_test, preds))
                    cv_metrics[mname]['precision'].append(precision_score(y_test, preds, average='weighted', zero_division=0))
                    cv_metrics[mname]['recall'].append(recall_score(y_test, preds, average='weighted', zero_division=0))
                    cv_metrics[mname]['f1'].append(f1_score(y_test, preds, average='weighted', zero_division=0))

                    if mname == 'KNN_CLAS':
                        cv_metrics[mname]['expert_count'].append(model.named_steps['classifier'].expert_X_.shape[0])
                    else:
                        cv_metrics[mname]['expert_count'].append(model.named_steps['classifier'].X_train_.shape[0])
                    
            wilcoxon_results = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                a = np.array(cv_metrics['KNN'][metric])
                b = np.array(cv_metrics['KNN_CLAS'][metric])
                differences = a - b
                if np.all(differences == 0):
                    wilcoxon_results[metric] = {'statistic': 0.0, 'p_value': 1.0}
                else:
                    stat, pval = wilcoxon(differences)
                    wilcoxon_results[metric] = {'statistic': stat, 'p_value': pval}

            ttest_rel_results = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                a = np.array(cv_metrics['KNN'][metric])
                b = np.array(cv_metrics['KNN_CLAS'][metric])
                stat, pval = ttest_rel(a, b)
                diff = a - b
                std_diff = diff.std(ddof=1)
                if std_diff == 0:
                    cohen_d = np.inf
                else:
                    cohen_d = diff.mean() / std_diff
                ttest_rel_results[metric] = {
                    'statistic': stat,
                    'p_value': pval,
                    'cohen_d': cohen_d
                }
            
            results[dname] = {
                'cv_metrics': cv_metrics,
                'statistical_tests': wilcoxon_results,
                'ttest_rel_results': ttest_rel_results
            }
            
        except Exception as e:
            print(f"Error processing {dname}: {str(e)}")
            results[dname] = {'error': str(e)}
    
    with open(os.path.join(output_dir, 'statistical_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def run_likelihood_analysis(datasets: dict, output_dir: str = 'output') -> None:
    spatial_results = {}
    
    preprocessor = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=1e-3)),
        ('correlation_filter', CorrelationFilter(threshold=0.9)),
        ('scaler', StandardScaler())
    ])

    for dname, data in datasets.items():
        print(f"\n=== Analyzing {dname} ===")
        X = data['X']
        y = data['y']
        
        try:
            knn_pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', KNN(k=5))
            ]).fit(X, y)

            knn_clas_pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', KNN_CLAS(k=5))
            ]).fit(X, y)

            X_transformed = knn_pipe.named_steps['preprocessor'].transform(X)
            
            q0_knn, q1_knn = knn_pipe.named_steps['classifier'].likelihood_score(X_transformed)
            q0_clas, q1_clas = knn_clas_pipe.named_steps['classifier'].likelihood_score(X_transformed)
            
            # Compute class masks using direct labels
            class0_mask = (y == -1)
            class1_mask = (y == 1)
            
            if not (np.any(class0_mask) and np.any(class1_mask)):
                raise ValueError("Both classes must be present for analysis")
            
            # 1. Centroids distance
            centroid0 = X_transformed[class0_mask].mean(axis=0)
            centroid1 = X_transformed[class1_mask].mean(axis=0)
            centroid_distance = float(np.linalg.norm(centroid0 - centroid1))
            
            # 2. Mean distance between points of different classes
            cross_dists = cdist(X_transformed[class0_mask], X_transformed[class1_mask], 'euclidean')
            mean_cross_dist = float(cross_dists.mean())
            
            # 3. Mean distance between points of the same class
            def compute_intra_stats(X_class):
                if len(X_class) < 2:
                    return 0.0, 0.0
                dists = pdist(X_class, 'euclidean')
                return float(dists.mean()), float(dists.var())
            
            intra_mean0, intra_var0 = compute_intra_stats(X_transformed[class0_mask])
            intra_mean1, intra_var1 = compute_intra_stats(X_transformed[class1_mask])
            mean_intra = (intra_mean0 + intra_mean1) / 2
            
            # 4. Mean distance to centroid
            def centroid_distances(X_class, centroid):
                if len(X_class) == 0:
                    return 0.0
                return np.linalg.norm(X_class - centroid, axis=1).mean()
            
            centroid_dist0 = centroid_distances(X_transformed[class0_mask], centroid0)
            centroid_dist1 = centroid_distances(X_transformed[class1_mask], centroid1)
            mean_centroid_dist = (centroid_dist0 + centroid_dist1) / 2
            
            # 5. Variance of intra-class distances
            var_intra = (intra_var0 + intra_var1) / 2
            
            # 6. Variance for Q0 and Q1 per class
            q_metrics = {}
            for model_name, q0, q1 in [('KNN', q0_knn, q1_knn), ('KNN_CLAS', q0_clas, q1_clas)]:
                for c in [-1, 1]:
                    mask = (y == c)
                    q0_vals = q0[mask]
                    q1_vals = q1[mask]
                    
                    var_q0 = float(np.var(q0_vals)) if len(q0_vals) >= 2 else None
                    var_q1 = float(np.var(q1_vals)) if len(q1_vals) >= 2 else None
                    
                    q_metrics[f'{model_name}_class{c}_var_q0'] = var_q0
                    q_metrics[f'{model_name}_class{c}_var_q1'] = var_q1
            
            # 8. Bhattacharyya's distance
            try:
                cov0 = np.cov(X_transformed[class0_mask], rowvar=False)
                cov1 = np.cov(X_transformed[class1_mask], rowvar=False)
                avg_cov = (cov0 + cov1) / 2
                mean_diff = centroid1 - centroid0
                
                term1 = 0.125 * mean_diff @ np.linalg.pinv(avg_cov) @ mean_diff
                det_avg = np.linalg.det(avg_cov)
                det_prod = np.sqrt(np.linalg.det(cov0) * np.linalg.det(cov1))
                term2 = 0.5 * np.log(det_avg / det_prod) if det_prod > 0 else np.nan
                bhatt_distance = float(term1 + term2) if not np.isnan(term2) else None
            except np.linalg.LinAlgError:
                bhatt_distance = None
            
            spatial_results[dname] = {
                'centroid_distance': centroid_distance,
                'mean_cross_class_distance': mean_cross_dist,
                'mean_intra_class_distance': mean_intra,
                'mean_centroid_distance': mean_centroid_dist,
                'variance_intra_class_distances': var_intra,
                'bhattacharyya_distance': bhatt_distance,
                **q_metrics
            }
            
        except Exception as e:
            print(f"Error analyzing {dname}: {str(e)}")
            spatial_results[dname] = {'error': str(e)}
    
    with open(os.path.join(output_dir, 'spatial_results.json'), 'w') as f:
        json.dump(spatial_results, f, indent=2)


if __name__ == '__main__':
    datasets = load_datasets()
    run_statistical_validation(datasets)
    run_likelihood_analysis(datasets)
    print("\n🎉 Analysis complete! Results saved to output/ directory")