import os
import time
import json
import numpy as np

from enum import IntEnum
from typing import Tuple, Self

from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import wilcoxon, ttest_rel

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

import ot

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
        self.label_map = {}
        self.inverse_map = {}

    def _validate_labels(self, y: np.ndarray) -> None:
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError("KNN currently only supports binary classification")
        self.label_map = {unique[0]: -1, unique[1]: 1}
        self.inverse_map = {-1: unique[0], 1: unique[1]}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_X_y(X, y)
        self._validate_labels(y)
        self.X_train_ = X
        self.y_train_ = np.vectorize(self.label_map.get)(y)
        self.cov_ = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.atleast_2d(X)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_)
        scores = np.sum(kernels * self.y_train_[nearest], axis=2)
        
        return np.vectorize(self.inverse_map.get)(np.where(scores.mean(1) > 0, 1, -1))

    def likelihood_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_)
        q0 = np.sum(kernels * (self.y_train_[nearest] == -1), axis=1)
        q1 = np.sum(kernels * (self.y_train_[nearest] == 1), axis=1)

        q_total = q0 + q1
        q0_norm = q0 / q_total
        q1_norm = q1 / q_total
        
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
        self.expert_y_ = np.vectorize(self.label_map.get)(self.expert_y_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_)
        scores = np.sum(kernels * self.expert_y_[nearest], axis=2)
        
        return np.vectorize(self.inverse_map.get)(np.where(scores.mean(1) > 0, 1, -1))
    
    def likelihood_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_)
        q0 = np.sum(kernels * (self.expert_y_[nearest] == -1), axis=1)
        q1 = np.sum(kernels * (self.expert_y_[nearest] == 1), axis=1)

        q_total = q0 + q1
        q0_norm = q0 / q_total
        q1_norm = q1 / q_total
        
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
        start_time = time.time()
        
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
                stat, pval = wilcoxon(cv_metrics['KNN'][metric], cv_metrics['KNN_CLAS'][metric])
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
                'ttest_rel_results': ttest_rel_results,
                'duration': time.time() - start_time
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
        start_time = time.time()
        
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
            
            centroid_knn = np.array([q0_knn.mean(), q1_knn.mean()])
            centroid_clas = np.array([q0_clas.mean(), q1_clas.mean()])
            centroid_dist = np.linalg.norm(centroid_knn - centroid_clas)
            M = cdist(np.column_stack([q0_knn, q1_knn]), np.column_stack([q0_clas, q1_clas]))
            emd = ot.emd2(np.ones(len(X))/len(X), np.ones(len(X))/len(X), M)
            
            spatial_results[dname] = {
                'centroid_distance': centroid_dist,
                'earth_movers_distance': emd,
                'duration': time.time() - start_time
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