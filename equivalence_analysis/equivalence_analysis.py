import os
import time
import json
import numpy as np
import networkx as nx

from enum import IntEnum

from typing import (
    Tuple, Self, Any, Dict, List, cast, Optional, TypedDict, NamedTuple
)

from numpy.typing import NDArray

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

k_values: List[int] = [1, 3, 5, 7]#, 11, 13, 17, 19, 23, 29]

class Adjacency(IntEnum):
    NOT_ADJACENT = 0
    GABRIEL_EDGE = 1
    SUPPORT_EDGE = 2

def vectorized_kernel(X: NDArray[np.float64], Y: NDArray[np.float64],
                        cov_inv: NDArray[np.float64], norm_factor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized Gaussian kernel computation."""
    X_diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    exponent = -0.5 * np.einsum('...i,ij,...j->...', X_diff, cov_inv, X_diff)
    return norm_factor * np.exp(exponent)

def gabriel_graph(dist_matrix: NDArray[np.float64]) -> NDArray[np.int64]:
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

def support_graph(gabriel_adj: NDArray[np.int64], y: NDArray[np.int64]) -> NDArray[np.int64]:
    """Create support graph from Gabriel graph and labels."""
    support_adj = np.zeros_like(gabriel_adj)
    diff_labels = y[:, None] != y[None, :]
    support_edges = (gabriel_adj > 0) & diff_labels
    support_adj[support_edges] = Adjacency.SUPPORT_EDGE
    support_adj[(gabriel_adj > 0) & ~diff_labels] = Adjacency.GABRIEL_EDGE
    return support_adj

class KNN(BaseEstimator, ClassifierMixin):
    X_train_: NDArray[np.float64]
    y_train_: NDArray[np.int64]

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Self:
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        n_features = X.shape[1]
        I_reg = 1e-6 * np.eye(n_features)
        cov = np.cov(X, rowvar=False) + I_reg
        self.cov_inv_ = np.linalg.pinv(cov)
        cov_det_ = np.linalg.det(cov)
        self.norm_factor_ = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det_)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        check_is_fitted(self)
        X = np.atleast_2d(X)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_inv_, self.norm_factor_)
        scores = np.sum(kernels * self.y_train_[nearest], axis=1).sum(axis=1)
        
        return np.where(scores > 0, 1, -1).astype(np.int64)

    def likelihood_score(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.X_train_[nearest], self.cov_inv_, self.norm_factor_)
        q0 = np.sum(kernels * (self.y_train_[nearest] == -1), axis=1).sum(axis=1)
        q1 = np.sum(kernels * (self.y_train_[nearest] == 1), axis=1).sum(axis=1)

        q_total = q0 + q1
        q_sum = np.sum(q_total)
        q0 = q0 / q_sum
        q1 = q1 / q_sum
        
        return q0, q1

class KNN_CLAS(KNN):
    expert_X_: NDArray[np.float64]
    expert_y_: NDArray[np.int64]

    def __init__(self, k: int = 3) -> None:
        super().__init__(k=k)

    def _get_experts(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        dist_matrix = squareform(pdist(X, metric='sqeuclidean'))
        gabriel_adj = gabriel_graph(dist_matrix)
        support_adj = support_graph(gabriel_adj, y)
        expert_mask = np.any(support_adj == Adjacency.SUPPORT_EDGE, axis=1)
        
        if not np.any(expert_mask):
            raise ValueError("No experts found in the dataset.")
            return X, y
        
        return X[expert_mask], y[expert_mask]

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Self:
        super().fit(X, y)
        self.expert_X_, self.expert_y_ = self._get_experts(X, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_inv_, self.norm_factor_)
        scores = np.sum(kernels * self.expert_y_[nearest], axis=1).sum(axis=1)
        
        return np.where(scores > 0, 1, -1).astype(np.int64)
    
    def likelihood_score(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self)
        pairwise_dists = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest = np.argpartition(pairwise_dists, self.k, axis=1)[:, :self.k]
        kernels = vectorized_kernel(X[:, np.newaxis], self.expert_X_[nearest], self.cov_inv_, self.norm_factor_)
        q0 = np.sum(kernels * (self.expert_y_[nearest] == -1), axis=1).sum(axis=1)
        q1 = np.sum(kernels * (self.expert_y_[nearest] == 1), axis=1).sum(axis=1)

        q_total = q0 + q1
        q_sum = np.sum(q_total)
        q0 = q0 / q_sum
        q1 = q1 / q_sum
        
        return q0, q1

class CorrelationFilter(BaseEstimator, TransformerMixin):
    to_drop_: NDArray[np.int64]
    threshold: float

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold
        self.to_drop_: NDArray[np.int64] = np.array([], dtype=int)

    def fit(self, X: NDArray[np.float64], y: Any = None) -> Self:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        corr = np.corrcoef(X, rowvar=False)
        upper_mask = np.triu(np.ones((n_features, n_features), bool), k=1)
        high_corr = (np.abs(corr) > self.threshold) & upper_mask
        self.to_drop_ = np.unique(np.where(high_corr)[1])

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=float)

        if X.shape[1] == 0 or self.to_drop_.size == 0:
            return X.copy()
        
        return np.delete(X, self.to_drop_, axis=1)

class DatasetType(TypedDict):
    X: NDArray[np.float64]
    y: NDArray[np.int64]

def load_datasets(data_dir: str = './sets') -> Dict[str, DatasetType]:
    datasets: Dict[str, DatasetType] = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('.npz'):
            try:
                data = np.load(os.path.join(data_dir, fname))
                name = fname.split('.')[0]
                X = data['X']
                y = data['y'].squeeze().astype(np.int64)
                datasets[name] = {'X': X.astype(np.float64), 'y': y}
                print(f"Loaded {name}: {X.shape}")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue
    return datasets

class WilcoxonResult(TypedDict):
    statistic: float
    p_value: float

class TTestResult(TypedDict):
    statistic: float
    p_value: float
    cohen_d: float

class KNNResults(TypedDict):
    avg_expert_count: Dict[str, int]
    wilcoxon: Dict[str, WilcoxonResult]
    ttest_rel: Dict[str, TTestResult]

class StatisticalResults(TypedDict):
    results: Dict[str, KNNResults]

class TTestRelResult(NamedTuple):
    statistic: float
    pvalue: float

def run_statistical_validation(datasets: Dict[str, DatasetType], output_dir: str = 'output') -> None:
    os.makedirs(output_dir, exist_ok=True)
    results: StatisticalResults = {'results': {}}
    
    preprocessor = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=1e-3)),
        ('correlation_filter', CorrelationFilter(threshold=0.9)),
        ('scaler', StandardScaler())
    ])

    print("\n=== Starting statistical validation ===")

    for dname, data in datasets.items():
        print(f"\n=== Processing {dname} ===")
        X = data['X']
        y = data['y']
        
        try:
            results[dname] = {}
            
            for k in k_values:
                print(f"--- Evaluating k={k} ---")
                cv_metrics: Dict[str, Dict[str, List[float]]] = {
                    'KNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []},
                    'KNN_CLAS': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []}
                }
                
                for train_idx, test_idx in KFold(n_splits=10).split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    X_train = preprocessor.fit_transform(X_train)
                    X_test = preprocessor.transform(X_test)
                    
                    for model in [KNN(k=k), KNN_CLAS(k=k)]:
                        mname = model.__class__.__name__
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        cv_metrics[mname]['accuracy'].append(float(accuracy_score(y_test, preds)))
                        cv_metrics[mname]['precision'].append(float(precision_score(y_test, preds, average='weighted', zero_division=0)))
                        cv_metrics[mname]['recall'].append(float(recall_score(y_test, preds, average='weighted', zero_division=0)))
                        cv_metrics[mname]['f1'].append(float(f1_score(y_test, preds, average='weighted', zero_division=0)))

                        if mname == 'KNN_CLAS':
                            expert_count = model.expert_X_.shape[0]
                        else:
                            expert_count = model.X_train_.shape[0]
                        cv_metrics[mname]['expert_count'].append(expert_count)

                avg_expert_count: Dict[str, int] = {}
                for name in ['KNN', 'KNN_CLAS']:
                    avg_expert_count[name] = int(np.mean(cv_metrics[name]['expert_count']))
                
                wilcoxon_results: Dict[str, WilcoxonResult] = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    a = np.array(cv_metrics['KNN'][metric])
                    b = np.array(cv_metrics['KNN_CLAS'][metric])
                    differences = a - b
                    if np.all(differences == 0):
                        wilcoxon_results[metric] = {'statistic': 0.0, 'p_value': 1.0}
                    else:
                        stat, pval = wilcoxon(differences)
                        wilcoxon_results[metric] = {'statistic': stat, 'p_value': pval}

                ttest_rel_results: Dict[str, TTestResult] = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    a = np.array(cv_metrics['KNN'][metric])
                    b = np.array(cv_metrics['KNN_CLAS'][metric])
                    diff = a - b

                    if len(diff) == 0:
                        stat = 0.0
                        pval = 1.0
                        cohen_d = 0.0
                    else:
                        all_same = np.all(diff == diff[0])
                        if all_same:
                            mean_diff = diff.mean()
                            if mean_diff == 0:
                                stat = 0.0
                                pval = 1.0
                                cohen_d = 0.0
                            else:
                                stat = 1e12 if mean_diff > 0 else -1e12
                                pval = 0.0
                                cohen_d = 1e12 if mean_diff > 0 else -1e12
                        else:
                            try:
                                ttest_rel_result = cast(TTestRelResult, ttest_rel(a, b))
                                stat = ttest_rel_result.statistic
                                pval = ttest_rel_result.pvalue
                            except:
                                stat = 0.0
                                pval = 1.0
                            mean_diff = diff.mean()
                            std_diff = diff.std(ddof=1)
                            if std_diff == 0:
                                cohen_d = 1e12 if mean_diff != 0 else 0.0
                            else:
                                cohen_d = mean_diff / std_diff

                    stat = stat if np.isfinite(stat) else (1e12 if stat > 0 else -1e12)
                    cohen_d = cohen_d if np.isfinite(cohen_d) else (1e12 if cohen_d > 0 else -1e12)

                    ttest_rel_results[metric] = {
                        'statistic': float(stat),
                        'p_value': float(pval),
                        'cohen_d': float(cohen_d)
                    }                          
                
                results[dname][str(k)] = {
                    'avg_expert_count': avg_expert_count,
                    'wilcoxon': wilcoxon_results,
                    'ttest_rel': ttest_rel_results
                }
            
        except Exception as e:
            print(f"Error processing {dname}: {str(e)}")
            results[dname] = {'error': str(e)}
    
    with open(os.path.join(output_dir, 'statistical_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Statistical validation completed! Results saved to output/statistical_results.json")

class SpatialMetrics(TypedDict):
    centroid_distance: float
    mean_distance_opposite: float
    mean_distance_same: float
    mean_dist_centroid: float
    bhattacharyya_distance: float
    var_dist_same: float
    var_dist_centroid: float
    var_q0: float
    var_q1: float

def run_likelihood_analysis(datasets: Dict[str, DatasetType], output_dir: str = 'output') -> None:
    spatial_results: Dict[str, Dict[str, Dict[str, SpatialMetrics]]] = {}
    
    print("\n=== Starting likelihood analysis ===")

    for dname, data in datasets.items():
        print(f"\n=== Processing {dname} ===")
        X = data['X']
        y = data['y']
        spatial_results[dname] = {}

        preprocessor = Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=1e-3)),
            ('correlation_filter', CorrelationFilter(threshold=0.9)),
            ('scaler', StandardScaler())
        ])

        X = preprocessor.fit_transform(X)
        
        for k in k_values:
            print(f"--- Evaluating k={k} ---")
            spatial_results[dname][str(k)] = {}
            
            for model in [KNN(k=k), KNN_CLAS(k=k)]:
                model_name = model.__class__.__name__

                try:
                    model.fit(X, y)
                    q0, q1 = model.likelihood_score(X)
                    
                    class_0_mask = (y == -1)
                    class_1_mask = (y == 1)
                    q0_class0 = q0[class_0_mask]
                    q1_class0 = q1[class_0_mask]
                    q0_class1 = q0[class_1_mask]
                    q1_class1 = q1[class_1_mask]
                    
                    points_class0 = np.column_stack((q0_class0, q1_class0)) if len(q0_class0) > 0 else np.empty((0, 2))
                    points_class1 = np.column_stack((q0_class1, q1_class1)) if len(q0_class1) > 0 else np.empty((0, 2))
                    
                    metrics: SpatialMetrics = {
                        'centroid_distance': 0.0,
                        'mean_distance_opposite': 0.0,
                        'mean_distance_same': 0.0,
                        'mean_dist_centroid': 0.0,
                        'bhattacharyya_distance': 0.0,
                        'var_dist_same': 0.0,
                        'var_dist_centroid': 0.0,
                        'var_q0': 0.0,
                        'var_q1': 0.0
                    }
            
                    # 1. Distance between centroids of opposite classes
                    centroid_class0: Optional[NDArray[np.float64]] = np.mean(points_class0, axis=0) if len(points_class0) > 0 else None
                    centroid_class1: Optional[NDArray[np.float64]] = np.mean(points_class1, axis=0) if len(points_class1) > 0 else None
                    if centroid_class0 is not None and centroid_class1 is not None:
                        centroid_distance = np.linalg.norm(centroid_class0 - centroid_class1)
                    else:
                        centroid_distance = 0.0
                    metrics['centroid_distance'] = cast(float, centroid_distance)
                    
                    # 2. Mean distance between samples from opposite classes
                    if len(points_class0) > 0 and len(points_class1) > 0:
                        distances_opposite = cdist(points_class0, points_class1, 'euclidean')
                        mean_distance_opposite = np.mean(distances_opposite)
                    else:
                        mean_distance_opposite = 0.0
                    metrics['mean_distance_opposite'] = cast(float, mean_distance_opposite)
                    
                    # 3. Mean distance between samples from same class
                    same_distances: List[float] = []
                    if len(points_class0) >= 2:
                        same_distances.append(cast(float, np.mean(pdist(points_class0, 'euclidean'))))
                    if len(points_class1) >= 2:
                        same_distances.append(cast(float, np.mean(pdist(points_class1, 'euclidean'))))
                    mean_distance_same = np.mean(same_distances) if same_distances else 0.0
                    metrics['mean_distance_same'] = cast(float, mean_distance_same)
                    
                    # 4. Mean distance between samples and their class centroid
                    centroid_dists: List[float] = []
                    if len(points_class0) > 0 and centroid_class0 is not None:
                        dists = np.linalg.norm(points_class0 - centroid_class0, axis=1)
                        centroid_dists.append(np.mean(dists) if len(dists) > 0 else 0.0)
                    if len(points_class1) > 0 and centroid_class1 is not None:
                        dists = np.linalg.norm(points_class1 - centroid_class1, axis=1)
                        centroid_dists.append(np.mean(dists) if len(dists) > 0 else 0.0)
                    mean_dist_centroid = np.mean(centroid_dists) if centroid_dists else 0.0
                    metrics['mean_dist_centroid'] = cast(float, mean_dist_centroid)
                    
                    # 5. Bhattacharyya distance
                    if len(points_class0) > 0 and len(points_class1) > 0 and centroid_class0 is not None and centroid_class1 is not None:
                        mu0 = centroid_class0
                        mu1 = centroid_class1
                        cov0 = np.cov(points_class0, rowvar=False)
                        cov1 = np.cov(points_class1, rowvar=False)
                        cov_avg = (cov0 + cov1) / 2
                        
                        diff_mu = mu0 - mu1
                        try:
                            inv_cov_avg = np.linalg.pinv(cov_avg)
                            term1 = 0.125 * diff_mu @ inv_cov_avg @ diff_mu.T
                        except:
                            term1 = 0.0
                        
                        # Compute log determinants using slogdet for numerical stability
                        sign_avg, logdet_avg = np.linalg.slogdet(cov_avg + 1e-9 * np.eye(cov_avg.shape[0]))
                        sign0, logdet0 = np.linalg.slogdet(cov0 + 1e-9 * np.eye(cov0.shape[0]))
                        sign1, logdet1 = np.linalg.slogdet(cov1 + 1e-9 * np.eye(cov1.shape[0]))
                        
                        if sign_avg <= 0 or sign0 <= 0 or sign1 <= 0:
                            term2 = 0.0
                        else:
                            # Calculate the log ratio directly
                            log_ratio = logdet_avg - 0.5 * (logdet0 + logdet1)
                            term2 = 0.5 * log_ratio
                        
                        bhattacharyya = term1 + term2
                    else:
                        bhattacharyya = 0.0
                    metrics['bhattacharyya_distance'] = cast(float, bhattacharyya)
                    
                    # 6. Variance of distances between same-class samples
                    same_variances: List[float] = []
                    if len(points_class0) >= 2:
                        same_variances.append(cast(float, np.var(pdist(points_class0, 'euclidean'))))
                    if len(points_class1) >= 2:
                        same_variances.append(cast(float, np.var(pdist(points_class1, 'euclidean'))))
                    var_dist_same = np.mean(same_variances) if same_variances else 0.0
                    metrics['var_dist_same'] = cast(float, var_dist_same)
                    
                    # 7. Variance of distances to centroid
                    centroid_variances: List[float] = []
                    if len(points_class0) > 0 and centroid_class0 is not None:
                        dists = np.linalg.norm(points_class0 - centroid_class0, axis=1)
                        centroid_variances.append(cast(float, np.var(dists) if len(dists) > 0 else 0.0))
                    if len(points_class1) > 0 and centroid_class1 is not None:
                        dists = np.linalg.norm(points_class1 - centroid_class1, axis=1)
                        centroid_variances.append(cast(float, np.var(dists) if len(dists) > 0 else 0.0))
                    var_dist_centroid = np.mean(centroid_variances) if centroid_variances else 0.0
                    metrics['var_dist_centroid'] = cast(float, var_dist_centroid)
                    
                    # 8. Variance for Q0 within same class
                    var_q0: float = 0.0
                    q0_vars: List[float] = []
                    if len(q0_class0) > 0:
                        q0_vars.append(cast(float, np.var(q0_class0)))
                    if len(q0_class1) > 0:
                        q0_vars.append(cast(float, np.var(q0_class1)))
                    var_q0 = cast(float, np.mean(q0_vars)) if q0_vars else 0.0
                    metrics['var_q0'] = var_q0
                    
                    # 9. Variance for Q1 within same class
                    var_q1: float = 0.0
                    q1_vars: List[float] = []
                    if len(q1_class0) > 0:
                        q1_vars.append(cast(float, np.var(q1_class0)))
                    if len(q1_class1) > 0:
                        q1_vars.append(cast(float, np.var(q1_class1)))
                    var_q1 = cast(float, np.mean(q1_vars)) if q1_vars else 0.0
                    metrics['var_q1'] = var_q1
            
                    spatial_results[dname][str(k)][model_name] = metrics

                except Exception as e:
                    print(f"Error in {model_name} (k={k}) on {dname}: {e}")
                    continue
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'spatial_results.json'), 'w') as f:
        json.dump(spatial_results, f, indent=2)

    print("\n✅ Likelihood analysis completed! Results saved to output/spatial_results.json")

if __name__ == '__main__':
    datasets = load_datasets()
    run_statistical_validation(datasets)
    run_likelihood_analysis(datasets)
    print("\n🎉 Analysis complete! Results saved to output/ directory")