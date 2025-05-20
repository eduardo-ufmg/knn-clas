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

k_values = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29]

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
        q0 = q0 / q_sum
        q1 = q1 / q_sum
        
        return q0, q1

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
        q0 = q0 / q_sum
        q1 = q1 / q_sum
        
        return q0, q1

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

    print("\n=== Starting statistical validation ===")

    for dname, data in datasets.items():
        print(f"\n=== Processing {dname} ===")
        X = data['X']
        y = data['y']
        
        try:
            results[dname] = {}
            
            for k in k_values:
                print(f"--- Evaluating k={k} ---")
                cv_metrics = {
                    'KNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []},
                    'KNN_CLAS': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'expert_count': []}
                }
                
                for train_idx, test_idx in KFold(n_splits=30).split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    for model in [
                            Pipeline([('preprocessor', preprocessor), ('classifier', KNN(k=k))]),
                            Pipeline([('preprocessor', preprocessor), ('classifier', KNN_CLAS(k=k))])
                        ]:
                        mname = model.named_steps['classifier'].__class__.__name__
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        cv_metrics[mname]['accuracy'].append(accuracy_score(y_test, preds))
                        cv_metrics[mname]['precision'].append(precision_score(y_test, preds, average='weighted', zero_division=0))
                        cv_metrics[mname]['recall'].append(recall_score(y_test, preds, average='weighted', zero_division=0))
                        cv_metrics[mname]['f1'].append(f1_score(y_test, preds, average='weighted', zero_division=0))

                        if mname == 'KNN_CLAS':
                            expert_count = model.named_steps['classifier'].expert_X_.shape[0]
                        else:
                            expert_count = model.named_steps['classifier'].X_train_.shape[0]
                        cv_metrics[mname]['expert_count'].append(expert_count)
                
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
                    cohen_d = diff.mean() / std_diff if std_diff != 0 else np.inf
                    ttest_rel_results[metric] = {'statistic': stat, 'p_value': pval, 'cohen_d': cohen_d}
                
                results[dname][str(k)] = {
                    'cv_metrics': cv_metrics,
                    'statistical_tests': wilcoxon_results,
                    'ttest_rel_results': ttest_rel_results
                }
            
        except Exception as e:
            print(f"Error processing {dname}: {str(e)}")
            results[dname] = {'error': str(e)}
    
    with open(os.path.join(output_dir, 'statistical_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Statistical validation completed! Results saved to output/statistical_results.json")


def run_likelihood_analysis(datasets: dict, output_dir: str = 'output') -> None:
    spatial_results = {}
    
    print("\n=== Starting likelihood analysis ===")

    for dname, data in datasets.items():
        print(f"\n=== Processing {dname} ===")
        X = data['X']
        y = data['y']
        spatial_results[dname] = {}
        
        for k in k_values:
            print(f"--- Evaluating k={k} ---")
            spatial_results[dname][str(k)] = {}
            
            for model_name in ['KNN', 'KNN_CLAS']:
                preprocessor = Pipeline([
                    ('variance_threshold', VarianceThreshold(threshold=1e-3)),
                    ('correlation_filter', CorrelationFilter(threshold=0.9)),
                    ('scaler', StandardScaler())
                ])
                
                if model_name == 'KNN':
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', KNN(k=k))
                    ])
                else:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', KNN_CLAS(k=k))
                    ])
                
                try:
                    model.fit(X, y)
                    X_transformed = model.named_steps['preprocessor'].transform(X)
                    classifier = model.named_steps['classifier']
                    q0, q1 = classifier.likelihood_score(X_transformed)
                    
                    class_0_mask = (y == -1)
                    class_1_mask = (y == 1)
                    q0_class0 = q0[class_0_mask]
                    q1_class0 = q1[class_0_mask]
                    q0_class1 = q0[class_1_mask]
                    q1_class1 = q1[class_1_mask]
                    
                    points_class0 = np.column_stack((q0_class0, q1_class0)) if len(q0_class0) > 0 else np.empty((0, 2))
                    points_class1 = np.column_stack((q0_class1, q1_class1)) if len(q0_class1) > 0 else np.empty((0, 2))
                    
                    metrics = {}
            
                    # 1. Distance between centroids of opposite classes
                    centroid_class0 = np.mean(points_class0, axis=0) if len(points_class0) > 0 else None
                    centroid_class1 = np.mean(points_class1, axis=0) if len(points_class1) > 0 else None
                    if centroid_class0 is not None and centroid_class1 is not None:
                        centroid_distance = np.linalg.norm(centroid_class0 - centroid_class1)
                    else:
                        centroid_distance = 0.0
                    metrics['centroid_distance'] = centroid_distance
                    
                    # 2. Mean distance between samples from opposite classes
                    if len(points_class0) > 0 and len(points_class1) > 0:
                        distances_opposite = cdist(points_class0, points_class1, 'euclidean')
                        mean_distance_opposite = np.mean(distances_opposite)
                    else:
                        mean_distance_opposite = 0.0
                    metrics['mean_distance_opposite'] = mean_distance_opposite
                    
                    # 3. Mean distance between samples from same class
                    same_distances = []
                    if len(points_class0) >= 2:
                        same_distances.append(np.mean(pdist(points_class0, 'euclidean')))
                    if len(points_class1) >= 2:
                        same_distances.append(np.mean(pdist(points_class1, 'euclidean')))
                    mean_distance_same = np.mean(same_distances) if same_distances else 0.0
                    metrics['mean_distance_same'] = mean_distance_same
                    
                    # 4. Mean distance between samples and their class centroid
                    centroid_dists = []
                    if len(points_class0) > 0:
                        dists = np.linalg.norm(points_class0 - centroid_class0, axis=1) if centroid_class0 is not None else []
                        centroid_dists.append(np.mean(dists) if len(dists) > 0 else 0.0)
                    if len(points_class1) > 0:
                        dists = np.linalg.norm(points_class1 - centroid_class1, axis=1) if centroid_class1 is not None else []
                        centroid_dists.append(np.mean(dists) if len(dists) > 0 else 0.0)
                    mean_dist_centroid = np.mean(centroid_dists) if centroid_dists else 0.0
                    metrics['mean_dist_centroid'] = mean_dist_centroid
                    
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
                    metrics['bhattacharyya_distance'] = bhattacharyya
                    
                    # 6. Variance of distances between same-class samples
                    same_variances = []
                    if len(points_class0) >= 2:
                        same_variances.append(np.var(pdist(points_class0, 'euclidean')))
                    if len(points_class1) >= 2:
                        same_variances.append(np.var(pdist(points_class1, 'euclidean')))
                    var_dist_same = np.mean(same_variances) if same_variances else 0.0
                    metrics['var_dist_same'] = var_dist_same
                    
                    # 7. Variance of distances to centroid
                    centroid_variances = []
                    if len(points_class0) > 0 and centroid_class0 is not None:
                        dists = np.linalg.norm(points_class0 - centroid_class0, axis=1)
                        centroid_variances.append(np.var(dists) if len(dists) > 0 else 0.0)
                    if len(points_class1) > 0 and centroid_class1 is not None:
                        dists = np.linalg.norm(points_class1 - centroid_class1, axis=1)
                        centroid_variances.append(np.var(dists) if len(dists) > 0 else 0.0)
                    var_dist_centroid = np.mean(centroid_variances) if centroid_variances else 0.0
                    metrics['var_dist_centroid'] = var_dist_centroid
                    
                    # 8. Variance for Q0 within same class
                    var_q0 = 0.0
                    q0_vars = []
                    if len(q0_class0) > 0:
                        q0_vars.append(np.var(q0_class0))
                    if len(q0_class1) > 0:
                        q0_vars.append(np.var(q0_class1))
                    var_q0 = np.mean(q0_vars) if q0_vars else 0.0
                    metrics['var_q0'] = var_q0
                    
                    # 9. Variance for Q1 within same class
                    var_q1 = 0.0
                    q1_vars = []
                    if len(q1_class0) > 0:
                        q1_vars.append(np.var(q1_class0))
                    if len(q1_class1) > 0:
                        q1_vars.append(np.var(q1_class1))
                    var_q1 = np.mean(q1_vars) if q1_vars else 0.0
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