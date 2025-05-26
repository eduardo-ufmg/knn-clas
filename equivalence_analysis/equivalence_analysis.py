import time
import json
import logging
import numpy as np

from enum import IntEnum
from pathlib import Path
from typing import (
    Tuple, Self, Any, Dict, List, Optional, TypedDict, Final
)

from numpy.typing import NDArray

from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import wilcoxon, ttest_rel # type: ignore[import-untyped]

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array # type: ignore[import-untyped]
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score # type: ignore[import-untyped]
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global seed for reproducibility
np.random.seed(0)

# K values for KNN experiments
K_VALUES: Final[List[int]] = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29]
H_VALUES: Final[List[int]] = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]

class Adjacency(IntEnum):
    """Enumeration for types of adjacency in graphs."""
    NOT_ADJACENT = 0
    GABRIEL_EDGE = 1
    SUPPORT_EDGE = 2

def gaussian_kernel_pairwise(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    cov_inv: NDArray[np.float64],
    norm_factor: float
) -> NDArray[np.float64]:
    """
    Computes Gaussian kernel values for all pairs of points between X and Y.
    X: (N, D) array of N points, D features.
    Y: (M, D) array of M points, D features.
    cov_inv: Inverse of the covariance matrix (D, D).
    norm_factor: Normalization factor for the Gaussian.
    Returns: (N, M) array of kernel values.
    """
    X_diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # Shape (N, M, D)
    exponent = -0.5 * np.einsum('NMi,ij,NMj->NM', X_diff, cov_inv, X_diff, optimize='optimal')
    return norm_factor * np.exp(exponent)

def batched_gaussian_kernels(
    X_points: NDArray[np.float64],
    Y_neighbor_sets: NDArray[np.float64],
    h: float,
    cov_inv_scaled: NDArray[np.float64],
    norm_factor: float
) -> NDArray[np.float64]:
    """
    Computes Gaussian kernel values between N points and their respective K neighbors.
    X_points: (N, D) array of N points.
    Y_neighbor_sets: (N, K, D) array where Y_neighbor_sets[i] are the K neighbors of X_points[i].
    h: Bandwidth for the Gaussian kernel.
    cov_inv_scaled: Inverse of the covariance matrix (D, D).
    norm_factor: Normalization factor for the Gaussian.
    Returns: (N, K) array of kernel values, where out[i,j] is K(X_points[i], Y_neighbor_sets[i,j]).
    """
    X_points_exp = X_points[:, np.newaxis, :]  # Shape (N, 1, D)
    # Y_neighbor_sets is (N, K, D)
    # Broadcasting X_points_exp (N,1,D) with Y_neighbor_sets (N,K,D)
    # results in X_diff (N,K,D) where X_diff[i,j,:] = X_points[i,:] - Y_neighbor_sets[i,j,:]
    X_diff = X_points_exp - Y_neighbor_sets / h  # Shape (N, K, D)
    
    # einsum for batched matrix multiplication: sum_d1 sum_d2 (X_diff[n,k,d1] * cov_inv[d1,d2] * X_diff[n,k,d2])
    exponent = -0.5 * np.einsum('NKi,ij,NKj->NK', X_diff, cov_inv_scaled, X_diff, optimize='optimal')
    return norm_factor * np.exp(exponent)

def gabriel_graph(dist_matrix_sq: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    Construct Gabriel graph from a pairwise squared distance matrix.
    An edge (i,j) exists if the hypersphere with diameter d(i,j)
    contains no other point k. This is equivalent to:
    d_sq(i,k) + d_sq(j,k) > d_sq(i,j) for all k != i,j.
    """
    n = dist_matrix_sq.shape[0]
    adjacency = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            d_sq_ij = dist_matrix_sq[i, j]
            is_gabriel_edge = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if dist_matrix_sq[i, k] + dist_matrix_sq[j, k] <= d_sq_ij:
                    is_gabriel_edge = False
                    break
            if is_gabriel_edge:
                adjacency[i, j] = Adjacency.GABRIEL_EDGE
                adjacency[j, i] = Adjacency.GABRIEL_EDGE
    return adjacency

def support_graph(gabriel_adj: NDArray[np.int64], y: NDArray[np.int64]) -> NDArray[np.int64]:
    """Create support graph from Gabriel graph and labels."""
    support_adj = np.zeros_like(gabriel_adj)
    # Create a boolean matrix: diff_labels[i,j] is True if y[i] != y[j]
    diff_labels = y[:, None] != y[None, :]
    
    # Support edges are Gabriel edges connecting points with different labels
    support_edges_mask = (gabriel_adj == Adjacency.GABRIEL_EDGE) & diff_labels
    support_adj[support_edges_mask] = Adjacency.SUPPORT_EDGE
    
    # Other Gabriel edges (connecting points with same labels) remain Gabriel edges
    same_label_gabriel_mask = (gabriel_adj == Adjacency.GABRIEL_EDGE) & ~diff_labels
    support_adj[same_label_gabriel_mask] = Adjacency.GABRIEL_EDGE
    
    return support_adj

class KNN(BaseEstimator, ClassifierMixin):
    """K-Nearest Neighbors classifier with Gaussian kernel using Mahalanobis distance."""
    X_train_: NDArray[np.float64]
    y_train_: NDArray[np.int64]
    cov_inv_: NDArray[np.float64]
    cov_inv_scaled_: NDArray[np.float64]
    norm_factor_: float

    def __init__(self, k: int = 3, h: float = 1.0) -> None:
        self.k = k
        self.h = h

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Self:
        X, y = check_X_y(X, y, ensure_2d=True, dtype=[np.float64, np.float32])
        self.X_train_ = X
        self.y_train_ = y
        
        n_features = X.shape[1]
        if n_features == 0:
            raise ValueError("Cannot fit KNN on 0 features.")

        # Add regularization to covariance matrix for stability
        I_reg = 1e-6 * np.eye(n_features)
        # Note: np.cov assumes rows are observations by default if rowvar=True (default)
        # If X samples are rows (N,D), then rowvar=False
        cov = np.cov(X, rowvar=False) + I_reg
        
        try:
            self.cov_inv_ = np.linalg.pinv(cov) # Use pseudo-inverse for robustness
            cov_det = np.linalg.det(cov)
            if cov_det <= 0: # Check for non-positive determinant
                logging.warning(f"Covariance matrix determinant is non-positive ({cov_det}). Using a fallback norm_factor.")
                # Fallback if determinant is problematic, though pinv might still work
                self.norm_factor_ = 1.0 
            else:
                self.norm_factor_ = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)

        except np.linalg.LinAlgError as e:
            logging.error(f"Failed to compute inverse or determinant of covariance matrix: {e}")
            # Fallback: use identity matrix for cov_inv and a simple norm_factor
            self.cov_inv_ = np.eye(n_features) 
            self.norm_factor_ = 1.0

        self.cov_inv_scaled_ = self.cov_inv_ / (self.h ** 2)  # Scale the covariance inverse by h^2
            
        return self

    def predict(self, X: NDArray[np.float64], k: Optional[int] = None, h: Optional[int] = None) -> NDArray[np.int64]:
        check_is_fitted(self, ['X_train_', 'y_train_', 'cov_inv_', 'norm_factor_'])
        X = check_array(X, ensure_2d=True, dtype=[np.float64, np.float32])

        current_k = k if k is not None else self.k
        if not isinstance(current_k, int) or current_k <= 0:
            raise ValueError(f"Number of neighbors k must be a positive integer, got {current_k}")
        if current_k > self.X_train_.shape[0]:
            logging.warning(f"k ({current_k}) is greater than the number of training samples ({self.X_train_.shape[0]}). Setting k to {self.X_train_.shape[0]}.")
            current_k = self.X_train_.shape[0]

        current_h = h if h is not None else self.h
        if not isinstance(current_h, (int, float)) or current_h <= 0:
            raise ValueError(f"Bandwidth h must be a positive number, got {current_h}")

        # Pairwise squared Euclidean distances
        pairwise_dists_sq = cdist(X, self.X_train_, metric='sqeuclidean')
        
        # Get indices of k nearest neighbors for each point in X
        # argpartition is faster than argsort if only k smallest are needed
        nearest_indices = np.argpartition(pairwise_dists_sq, current_k -1 , axis=1)[:, :current_k]
        
        k_nearest_neighbors = self.X_train_[nearest_indices] # Shape (N_X_test, current_k, N_features)
        k_nearest_labels = self.y_train_[nearest_indices]    # Shape (N_X_test, current_k)

        kernels = batched_gaussian_kernels(X, k_nearest_neighbors, current_h, self.cov_inv_scaled_, self.norm_factor_) # Shape (N_X_test, current_k)
        
        # Weighted sum of labels of k nearest neighbors
        # scores[i] = sum_{j=0 to k-1} kernels[i,j] * k_nearest_labels[i,j]
        scores = np.sum(kernels * k_nearest_labels, axis=1) # Shape (N_X_test,)
        
        return np.where(scores >= 0, 1, -1).astype(np.int64) # Predict class 1 if score is non-negative

    def likelihood_score(self, X: NDArray[np.float64], k: Optional[int] = None, h: Optional[int] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self, ['X_train_', 'y_train_', 'cov_inv_', 'norm_factor_'])
        X = check_array(X, ensure_2d=True, dtype=[np.float64, np.float32])

        current_k = k if k is not None else self.k
        if not isinstance(current_k, int) or current_k <= 0:
            raise ValueError(f"Number of neighbors k must be a positive integer, got {current_k}")
        if current_k > self.X_train_.shape[0]:
            logging.warning(f"k ({current_k}) in likelihood_score is greater than training samples. Adjusting k.")
            current_k = self.X_train_.shape[0]
            
        current_h = h if h is not None else self.h
        if not isinstance(current_h, (int, float)) or current_h <= 0:
            raise ValueError(f"Bandwidth h must be a positive number, got {current_h}")

        pairwise_dists_sq = cdist(X, self.X_train_, metric='sqeuclidean')
        nearest_indices = np.argpartition(pairwise_dists_sq, current_k - 1, axis=1)[:, :current_k]
        
        k_nearest_neighbors = self.X_train_[nearest_indices]
        k_nearest_labels = self.y_train_[nearest_indices]
        
        kernels = batched_gaussian_kernels(X, k_nearest_neighbors, current_h, self.cov_inv_scaled_, self.norm_factor_) # (N_X_test, k)

        q0 = np.sum(kernels * (k_nearest_labels == -1), axis=1) # Sum of kernels for class -1 neighbors
        q1 = np.sum(kernels * (k_nearest_labels == 1), axis=1)  # Sum of kernels for class  1 neighbors

        return q0, q1

class KNN_CLAS(KNN):
    """KNN classifier variant that uses 'expert' support points for prediction."""
    expert_X_: NDArray[np.float64]
    expert_y_: NDArray[np.int64]

    def __init__(self, k: int = 3, h: float = 1.0) -> None:
        super().__init__(k=k, h=h)

    def _get_experts(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Identifies expert points based on Gabriel graph and support edges."""
        if X.shape[0] < 2:
             raise ValueError("Not enough samples to build Gabriel graph for expert selection.")
        dist_matrix_sq = squareform(pdist(X, metric='sqeuclidean'))
        gabriel_adj = gabriel_graph(dist_matrix_sq)
        support_adj = support_graph(gabriel_adj, y)
        
        # Experts are points that are part of at least one support edge
        expert_mask = np.any(support_adj == Adjacency.SUPPORT_EDGE, axis=1)
        
        if not np.any(expert_mask):
            raise ValueError("No experts (support points) found in the dataset. KNN_CLAS cannot proceed.")
        
        return X[expert_mask], y[expert_mask]

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Self:
        # Fit normally first to get cov_inv_ and norm_factor_ from all data
        super().fit(X, y) 
        # Then identify experts from the full training set
        self.expert_X_, self.expert_y_ = self._get_experts(self.X_train_, self.y_train_)
        
        if self.expert_X_.shape[0] == 0:
            # This case should be caught by _get_experts raising an error.
            # If _get_experts were to return empty arrays on failure instead of raising:
            raise ValueError("Expert selection resulted in zero experts. KNN_CLAS cannot be fitted.")
        
        # Ensure k is not greater than the number of experts
        if self.k > self.expert_X_.shape[0] and self.expert_X_.shape[0] > 0:
            logging.warning(
                f"Initial k ({self.k}) for KNN_CLAS is greater than the number of experts "
                f"({self.expert_X_.shape[0]}). Adjusting k to {self.expert_X_.shape[0]}."
            )
            self.k = self.expert_X_.shape[0]
        elif self.expert_X_.shape[0] == 0 : # Should not happen if _get_experts raises error
             self.k = 1 # Default k if somehow no experts and no error
        return self

    def predict(self, X: NDArray[np.float64], k: Optional[int] = None, h: Optional[int] = None) -> NDArray[np.int64]:
        check_is_fitted(self, ['expert_X_', 'expert_y_', 'cov_inv_', 'norm_factor_'])
        X = check_array(X, ensure_2d=True, dtype=[np.float64, np.float32])

        current_k = k if k is not None else self.k
        if not isinstance(current_k, int) or current_k <= 0:
            raise ValueError(f"Number of neighbors k must be a positive integer, got {current_k}")
        
        current_h = h if h is not None else self.h
        if not isinstance(current_h, (int, float)) or current_h <= 0:
            raise ValueError(f"Bandwidth h must be a positive number, got {current_h}")

        if self.expert_X_.shape[0] == 0:
             raise RuntimeError("KNN_CLAS has no expert points to predict from. Ensure fit was successful.")
        if current_k > self.expert_X_.shape[0]:
            logging.warning(f"k ({current_k}) for KNN_CLAS predict is greater than number of experts ({self.expert_X_.shape[0]}). Adjusting k.")
            current_k = self.expert_X_.shape[0]
        
        pairwise_dists_sq = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest_indices = np.argpartition(pairwise_dists_sq, current_k - 1, axis=1)[:, :current_k]
        
        k_nearest_expert_neighbors = self.expert_X_[nearest_indices]
        k_nearest_expert_labels = self.expert_y_[nearest_indices]
        
        # Use cov_inv_ and norm_factor_ from the original full dataset fit
        kernels = batched_gaussian_kernels(X, k_nearest_expert_neighbors, current_h, self.cov_inv_scaled_, self.norm_factor_)
        scores = np.sum(kernels * k_nearest_expert_labels, axis=1)
        
        return np.where(scores >= 0, 1, -1).astype(np.int64)
    
    def likelihood_score(self, X: NDArray[np.float64], k: Optional[int] = None, h: Optional[int] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self, ['expert_X_', 'expert_y_', 'cov_inv_', 'norm_factor_'])
        X = check_array(X, ensure_2d=True, dtype=[np.float64, np.float32])

        current_k = k if k is not None else self.k
        if not isinstance(current_k, int) or current_k <= 0:
            raise ValueError(f"Number of neighbors k must be a positive integer, got {current_k}")
        
        current_h = h if h is not None else self.h
        if not isinstance(current_h, (int, float)) or current_h <= 0:
            raise ValueError(f"Bandwidth h must be a positive number, got {current_h}")

        if self.expert_X_.shape[0] == 0:
             raise RuntimeError("KNN_CLAS has no expert points for likelihood_score. Ensure fit was successful.")
        if current_k > self.expert_X_.shape[0]:
            logging.warning(f"k ({current_k}) for KNN_CLAS likelihood is greater than number of experts ({self.expert_X_.shape[0]}). Adjusting k.")
            current_k = self.expert_X_.shape[0]

        pairwise_dists_sq = cdist(X, self.expert_X_, metric='sqeuclidean')
        nearest_indices = np.argpartition(pairwise_dists_sq, current_k -1, axis=1)[:, :current_k]
        
        k_nearest_expert_neighbors = self.expert_X_[nearest_indices]
        k_nearest_expert_labels = self.expert_y_[nearest_indices]

        kernels = batched_gaussian_kernels(X, k_nearest_expert_neighbors, current_h, self.cov_inv_scaled_, self.norm_factor_)
        
        q0 = np.sum(kernels * (k_nearest_expert_labels == -1), axis=1)
        q1 = np.sum(kernels * (k_nearest_expert_labels == 1), axis=1)
        
        return q0, q1

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Transformer to remove highly correlated features."""
    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold
        self.to_drop_: Optional[NDArray[np.int64]] = None # Features to drop

    def fit(self, X: NDArray[np.float64], y: Any = None) -> Self:
        X = check_array(X, dtype=[np.float64, np.float32])
        if X.shape[1] < 2:
            self.to_drop_ = np.array([], dtype=int)
            return self # Not enough features to compare correlations

        corr_matrix = np.corrcoef(X, rowvar=False)
        # Ensure corr_matrix is 2D, even if X has only one feature (np.corrcoef might return scalar)
        corr_matrix = np.atleast_2d(corr_matrix)

        upper_triangle_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        highly_correlated_pairs = (np.abs(corr_matrix) > self.threshold) & upper_triangle_mask
        
        # Get indices of columns to drop (prefer dropping the second feature in a pair)
        self.to_drop_ = np.unique(np.where(highly_correlated_pairs)[1]).astype(np.int64)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, 'to_drop_')
        X = check_array(X, dtype=[np.float64, np.float32])

        if self.to_drop_ is None or self.to_drop_.size == 0:
            return X.copy()
        
        # Ensure to_drop_ indices are valid for X's shape
        valid_to_drop = self.to_drop_[self.to_drop_ < X.shape[1]]
        if len(valid_to_drop) < len(self.to_drop_):
            logging.warning("CorrelationFilter: Some indices to drop were out of bounds for the input X.")

        return np.delete(X, valid_to_drop, axis=1)

# --- TypedDicts for data structures ---
class Dataset(TypedDict):
    X: NDArray[np.float64]
    y: NDArray[np.int64]

class WilcoxonResult(TypedDict):
    statistic: float
    pvalue: float

class TTestResult(TypedDict):
    statistic: float
    pvalue: float
    cohen_d: float

class KFoldMetrics(TypedDict): # Metrics for a specific k value over folds
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    f1: List[float]

class ModelFoldResults(TypedDict): # Results for KNN and KNN_CLAS for a given k
    KNN: KFoldMetrics
    KNN_CLAS: KFoldMetrics

class DatasetStatisticalResults(TypedDict): # Statistical test results for one dataset and one k
    wilcoxon: Dict[str, WilcoxonResult] # Metric name -> Wilcoxon result
    ttest_rel: Dict[str, TTestResult]  # Metric name -> T-test result

# Overall structure for statistical results
# DatasetName -> K_value -> h_value -> TestName -> MetricName -> ResultValue
StatisticalResults = Dict[str, Dict[int, Dict[float, DatasetStatisticalResults]]]

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

# Overall structure for spatial results
# DatasetName -> K_value -> h_value -> ModelName -> SpatialMetrics
SpatialAnalysisResults = Dict[str, Dict[int, Dict[float, Dict[str, SpatialMetrics]]]]


def load_datasets(data_dir: Path = Path('./sets')) -> Dict[str, Dataset]:
    """Loads datasets from .npz files in the specified directory."""
    datasets: Dict[str, Dataset] = {}
    if not data_dir.is_dir():
        logging.error(f"Data directory {data_dir} not found.")
        return datasets
        
    for fpath in data_dir.glob('*.npz'):
        try:
            data = np.load(fpath)
            name = fpath.stem # Get filename without extension
            
            if 'X' not in data or 'y' not in data:
                logging.warning(f"Skipping {fpath.name}: 'X' or 'y' key not found in npz file.")
                continue

            X = data['X']
            y = data['y'].squeeze().astype(np.int64) # Ensure y is 1D and integer
            
            # Basic validation of X and y
            if X.ndim != 2:
                logging.warning(f"Skipping {name}: X is not 2-dimensional (shape: {X.shape}).")
                continue
            if y.ndim != 1:
                logging.warning(f"Skipping {name}: y is not 1-dimensional after squeeze (shape: {y.shape}).")
                continue
            if X.shape[0] != y.shape[0]:
                logging.warning(f"Skipping {name}: X and y have inconsistent number of samples ({X.shape[0]} vs {y.shape[0]}).")
                continue
            if X.shape[0] == 0:
                logging.warning(f"Skipping {name}: Dataset is empty.")
                continue


            datasets[name] = {'X': X.astype(np.float64), 'y': y}
            logging.info(f"Loaded dataset '{name}': X shape {X.shape}, y shape {y.shape}")
        except Exception as e:
            logging.error(f"Error loading {fpath.name}: {e}")
            continue
    return datasets

def run_statistical_validation(datasets: Dict[str, Dataset], output_dir: Path = Path('output')) -> None:
    """Performs statistical validation of KNN vs KNN_CLAS using cross-validation."""
    logging.info("\n=== Starting statistical validation ===")

    preprocessor = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=1e-3)),
        ('correlation_filter', CorrelationFilter(threshold=0.9)), # Custom filter
        ('scaler', StandardScaler())
    ])

    # Initialize results structure
    all_results: StatisticalResults = {dname: {} for dname in datasets.keys()}
    metric_names = ['accuracy', 'precision', 'recall', 'f1']

    for dname, dataset_content in datasets.items():
        logging.info(f"\nProcessing dataset: {dname}")
        X, y = dataset_content['X'], dataset_content['y']

        # Store metrics from each fold for each k and model
        # cv_metrics: K_value -> h_value -> ModelName -> MetricName -> List_of_scores_from_folds
        cv_metrics_all: Dict[int, ModelFoldResults] = {
            k_val: {
                h_val: {
                    'KNN': {metric: [] for metric in metric_names},
                    'KNN_CLAS': {metric: [] for metric in metric_names}
                } for h_val in H_VALUES
            } for k_val in K_VALUES
        }
        
        if X.shape[0] < 2 : # KFold needs at least 2 samples
            logging.warning(f"Skipping dataset {dname} for statistical validation: not enough samples ({X.shape[0]}).")
            continue
        
        n_splits = min(10, X.shape[0]) # Adjust n_splits if fewer samples than 10
        if n_splits < 2:
            logging.warning(f"Skipping dataset {dname} for statistical validation: not enough samples for {n_splits}-Fold CV.")
            continue
            
        kf = KFold(n_splits=n_splits, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            logging.info(f"  Processing {dname}, Fold {fold_idx+1}/{n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                X_train_processed = preprocessor.fit_transform(X_train, y_train)
                X_test_processed = preprocessor.transform(X_test)
            except ValueError as e:
                logging.error(f"Error during preprocessing for {dname}, Fold {fold_idx+1}: {e}. Skipping fold.")
                continue


            models_to_test = {'KNN': KNN(), 'KNN_CLAS': KNN_CLAS()}
            for model_name, model_instance in models_to_test.items():
                try:
                    model_instance.fit(X_train_processed, y_train)
                    for k_val in K_VALUES:
                        # Adjust k if it's larger than available training/expert points
                        actual_k = k_val
                        if model_name == 'KNN':
                            if k_val > model_instance.X_train_.shape[0] and model_instance.X_train_.shape[0]>0 :
                                actual_k = model_instance.X_train_.shape[0]
                        elif model_name == 'KNN_CLAS':
                             if k_val > model_instance.expert_X_.shape[0] and model_instance.expert_X_.shape[0]>0:
                                actual_k = model_instance.expert_X_.shape[0]
                        
                        if actual_k == 0: # No points to predict from
                            logging.warning(f"Model {model_name} with k={k_val} has no points for prediction in {dname}, Fold {fold_idx+1}. Scoring as 0.")
                            for metric in metric_names:
                                cv_metrics_all[k_val][model_name][metric].append(0.0) # type: ignore
                            continue

                        for h_val in H_VALUES:

                            y_pred = model_instance.predict(X_test_processed, k=actual_k, h=h_val)
                            
                            # Ensure y_test and y_pred are not empty and have consistent labels for scoring
                            if len(y_test) == 0 or len(y_pred) == 0:
                                for metric in metric_names:
                                    cv_metrics_all[k_val][h_val][model_name][metric].append(0.0) # type: ignore
                                continue

                            # Calculate metrics
                            cv_metrics_all[k_val][h_val][model_name]['accuracy'].append(float(accuracy_score(y_test, y_pred))) #type: ignore
                            cv_metrics_all[k_val][h_val][model_name]['precision'].append(float(precision_score(y_test, y_pred, zero_division=0, labels=np.unique(y_test)))) #type: ignore
                            cv_metrics_all[k_val][h_val][model_name]['recall'].append(float(recall_score(y_test, y_pred, zero_division=0, labels=np.unique(y_test)))) #type: ignore
                            cv_metrics_all[k_val][h_val][model_name]['f1'].append(float(f1_score(y_test, y_pred, zero_division=0, labels=np.unique(y_test)))) #type: ignore

                except ValueError as e: # Catch errors from fit/predict (e.g. no experts)
                    logging.error(f"Error with model {model_name} for {dname}, Fold {fold_idx+1}: {e}. Scoring as 0 for this model in this fold.")
                    for k_val_err in K_VALUES:
                        for h_val_err in H_VALUES:
                            for metric in metric_names:
                                cv_metrics_all[k_val_err][h_val_err][model_name][metric].append(0.0) #type: ignore
                    # Continue to next model or fold
                except Exception as e: # Catch any other unexpected error
                    logging.critical(f"Unexpected error with model {model_name} for {dname}, Fold {fold_idx+1}: {e}")
                    # Decide if to skip fold, dataset, or stop
                    for k_val_err in K_VALUES:
                        for h_val_err in H_VALUES:
                            for metric in metric_names:
                                cv_metrics_all[k_val_err][h_val_err][model_name][metric].append(0.0) #type: ignore

        # Perform statistical tests for each k
        for k_val in K_VALUES:
            all_results[dname][k_val] = {}
            for h_val in H_VALUES:
                all_results[dname][k_val][h_val] = {'wilcoxon': {}, 'ttest_rel': {}} #type: ignore
                
                for metric in metric_names:
                    knn_scores = np.array(cv_metrics_all[k_val][h_val]['KNN'][metric]) #type: ignore
                    knn_clas_scores = np.array(cv_metrics_all[k_val][h_val]['KNN_CLAS'][metric]) #type: ignore

                    # Ensure there are scores to compare
                    if len(knn_scores) == 0 or len(knn_clas_scores) == 0 or len(knn_scores) != len(knn_clas_scores) :
                        logging.warning(f"Skipping stat tests for {dname}, k={k_val}, h={h_val}, metric={metric} due to insufficient/mismatched fold data.")
                        all_results[dname][k_val][h_val]['wilcoxon'][metric] = {'statistic': np.nan, 'pvalue': np.nan} #type: ignore
                        all_results[dname][k_val][h_val]['ttest_rel'][metric] = {'statistic': np.nan, 'pvalue': np.nan, 'cohen_d': np.nan} #type: ignore
                        continue

                    # Wilcoxon test
                    diff = knn_scores - knn_clas_scores
                    if np.all(np.abs(diff) < 1e-9): # Effectively all zero differences
                        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                    else:
                        try:
                            wilcoxon_stat, wilcoxon_p = wilcoxon(knn_scores, knn_clas_scores)
                        except ValueError as e: # e.g. too few samples, all differences are zero after internal processing
                            logging.warning(f"Wilcoxon test failed for {dname}, k={k_val}, h={h_val}, metric={metric}: {e}. Assigning default values.")
                            wilcoxon_stat, wilcoxon_p = (0.0, 1.0) if np.all(np.abs(diff) < 1e-9) else (np.nan, np.nan)

                    all_results[dname][k_val][h_val]['wilcoxon'][metric] = {'statistic': float(wilcoxon_stat), 'pvalue': float(wilcoxon_p)} #type: ignore

                    # Paired T-test
                    ttest_stat, ttest_p, cohen_d_val = np.nan, np.nan, np.nan
                    if np.all(np.abs(diff) < 1e-9): # All differences are zero
                        ttest_stat, ttest_p, cohen_d_val = 0.0, 1.0, 0.0
                    else:
                        # Check if all differences are identical (std_dev of diff will be 0)
                        if np.std(diff, ddof=1) < 1e-9: # Effectively zero standard deviation
                            mean_diff = np.mean(diff)
                            # If mean_diff is also zero, it's covered above. If non-zero, t-stat is undefined (inf).
                            ttest_stat = np.inf * np.sign(mean_diff) if mean_diff != 0 else 0.0
                            ttest_p = 0.0 if mean_diff != 0 else 1.0
                            cohen_d_val = np.inf * np.sign(mean_diff) if mean_diff != 0 else 0.0
                        else:
                            try:
                                ttest_stat, ttest_p = ttest_rel(knn_scores, knn_clas_scores)
                                mean_diff = np.mean(diff)
                                std_diff = np.std(diff, ddof=1) # Sample std dev of differences
                                cohen_d_val = mean_diff / std_diff if std_diff > 1e-9 else (np.inf * np.sign(mean_diff) if mean_diff !=0 else 0.0)
                            except Exception as e: # Catch any error during t-test
                                logging.warning(f"Paired t-test failed for {dname}, k={k_val}, h={h_val}, metric={metric}: {e}. Assigning NaN.")
                                # ttest_stat, ttest_p, cohen_d_val remain NaN

                    all_results[dname][k_val][h_val]['ttest_rel'][metric] = { #type: ignore
                        'statistic': float(np.nan_to_num(ttest_stat, nan=0.0, posinf=1e12, neginf=-1e12)), # Replace non-finite with large numbers
                        'pvalue': float(np.nan_to_num(ttest_p, nan=1.0)),
                        'cohen_d': float(np.nan_to_num(cohen_d_val, nan=0.0, posinf=1e12, neginf=-1e12))
                    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'statistical_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, allow_nan=False) # disallow_nan for strict JSON
        logging.info(f"✅ Statistical validation completed! Results saved to {results_file}")
    except TypeError as e: # Handle NaN if allow_nan=False and NaNs are present
        logging.error(f"Could not serialize statistical results to JSON (possibly due to NaN/inf values not handled by np.nan_to_num as expected): {e}")
        # Fallback: try saving with allow_nan=True (produces non-standard JSON but saves data)
        try:
            with open(output_dir / 'statistical_results_nonstandard.json', 'w') as f:
                json.dump(all_results, f, indent=2) 
            logging.warning("Saved statistical results to statistical_results_nonstandard.json with NaNs/Infs.")
        except Exception as e_fallback:
            logging.error(f"Fallback saving also failed: {e_fallback}")


def run_likelihood_analysis(datasets: Dict[str, Dataset], output_dir: Path = Path('output')) -> None:
    """Performs spatial likelihood analysis on KNN and KNN_CLAS outputs."""
    logging.info("\n=== Starting likelihood analysis ===")
    
    preprocessor = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=1e-3)),
        ('correlation_filter', CorrelationFilter(threshold=0.9)),
        ('scaler', StandardScaler())
    ])

    all_spatial_results: SpatialAnalysisResults = {dname: {} for dname in datasets.keys()}

    for dname, dataset_content in datasets.items():
        logging.info(f"\nProcessing dataset for likelihood analysis: {dname}")
        X_orig, y_orig = dataset_content['X'], dataset_content['y']

        if X_orig.shape[0] < 2 or X_orig.shape[1] == 0:
            logging.warning(f"Skipping likelihood analysis for {dname}: insufficient samples or features.")
            continue
        
        try:
            X_processed = preprocessor.fit_transform(X_orig, y_orig)
        except ValueError as e:
            logging.error(f"Error during preprocessing for likelihood analysis of {dname}: {e}. Skipping dataset.")
            continue

        models_for_likelihood: Dict[str, KNN] = {}
        try:
            models_for_likelihood['KNN'] = KNN().fit(X_processed, y_orig)
            models_for_likelihood['KNN_CLAS'] = KNN_CLAS().fit(X_processed, y_orig)
        except ValueError as e: # e.g. no experts for KNN_CLAS
            logging.error(f"Could not fit models for likelihood analysis on {dname}: {e}. Skipping dataset.")
            continue
        except Exception as e: # Catch any other unexpected error
            logging.critical(f"Unexpected error fitting models for likelihood analysis on {dname}: {e}")
            continue


        all_spatial_results[dname] = {k_val: {} for k_val in K_VALUES} #type: ignore

        for k_val in K_VALUES:
            all_spatial_results[dname][k_val] = {} #type: ignore

            for h_val in H_VALUES:
                all_spatial_results[dname][k_val][h_val] = {model_name: {} for model_name in models_for_likelihood.keys()} # type: ignore
                for model_name, model_instance in models_for_likelihood.items():
                    try:
                        q0, q1 = model_instance.likelihood_score(X_processed, k=k_val, h=h_val)
                    except RuntimeError as e: # E.g. KNN_CLAS predict called with no experts
                        logging.error(f"Runtime error during likelihood score for {model_name}, k={k_val}, h={h_val} on {dname}: {e}. Skipping this model.")
                        all_spatial_results[dname][k_val][h_val][model_name] = { # type: ignore
                                metric: np.nan for metric in SpatialMetrics.__annotations__
                            }
                        continue


                    class_0_mask = (y_orig == -1)
                    class_1_mask = (y_orig == 1)
                    
                    # Ensure there are points in each class for q value separation
                    if not (np.any(class_0_mask) and np.any(class_1_mask)):
                        logging.warning(f"Dataset {dname} does not have samples from both classes. Spatial metrics may be ill-defined.")
                        # Fill with NaNs or skip, depending on desired behavior
                        all_spatial_results[dname][k_val][h_val][model_name] = { # type: ignore
                                metric: np.nan for metric in SpatialMetrics.__annotations__
                            }
                        continue


                    q0_c0, q1_c0 = q0[class_0_mask], q1[class_0_mask]
                    q0_c1, q1_c1 = q0[class_1_mask], q1[class_1_mask]
                    
                    # (q0, q1) points for each class
                    points_c0 = np.column_stack((q0_c0, q1_c0)) if q0_c0.size > 0 else np.empty((0,2))
                    points_c1 = np.column_stack((q0_c1, q1_c1)) if q0_c1.size > 0 else np.empty((0,2))

                    metrics: SpatialMetrics = {key: np.nan for key in SpatialMetrics.__annotations__} # Initialize with NaN

                    # Centroids and their distance
                    if points_c0.shape[0] > 0 and points_c1.shape[0] > 0:
                        centroid_c0 = np.mean(points_c0, axis=0)
                        centroid_c1 = np.mean(points_c1, axis=0)
                        metrics['centroid_distance'] = float(np.linalg.norm(centroid_c0 - centroid_c1))
                        
                        # Mean distance between points from opposite classes
                        metrics['mean_distance_opposite'] = float(np.mean(cdist(points_c0, points_c1, metric='euclidean')))
                        
                        # Bhattacharyya Distance
                        # Requires >1 point per class to compute covariance
                        if points_c0.shape[0] > 1 and points_c1.shape[0] > 1:
                            cov_c0 = np.cov(points_c0, rowvar=False)
                            cov_c1 = np.cov(points_c1, rowvar=False)
                            
                            # Add small epsilon for numerical stability
                            eps = 1e-9 * np.eye(points_c0.shape[1]) 
                            cov_c0 = cov_c0 + eps
                            cov_c1 = cov_c1 + eps
                            
                            cov_avg = (cov_c0 + cov_c1) / 2.0
                            diff_mu = centroid_c0 - centroid_c1
                            
                            try:
                                inv_cov_avg = np.linalg.inv(cov_avg)
                                term1 = 0.125 * diff_mu @ inv_cov_avg @ diff_mu.T
                                
                                slogdet_cov_avg = np.linalg.slogdet(cov_avg)
                                slogdet_cov_c0 = np.linalg.slogdet(cov_c0)
                                slogdet_cov_c1 = np.linalg.slogdet(cov_c1)

                                if slogdet_cov_avg[0] > 0 and slogdet_cov_c0[0] > 0 and slogdet_cov_c1[0] > 0: # Valid log-determinants
                                    term2 = 0.5 * (slogdet_cov_avg[1] - 0.5 * (slogdet_cov_c0[1] + slogdet_cov_c1[1]))
                                    metrics['bhattacharyya_distance'] = float(term1 + term2)
                                else:
                                    logging.debug(f"Skipping Bhattacharyya term2 due to non-positive determinant for {dname}, k={k_val}, model={model_name}")
                                    metrics['bhattacharyya_distance'] = float(term1) # Or NaN if term2 is crucial
                            except np.linalg.LinAlgError:
                                logging.warning(f"LinAlgError in Bhattacharyya calculation for {dname}, k={k_val}, model={model_name}.")
                                # metrics['bhattacharyya_distance'] remains NaN or set to a specific error indicator
                    
                    # Mean distance within the same class
                    same_distances: List[float] = []
                    if points_c0.shape[0] > 1: same_distances.extend(pdist(points_c0, metric='euclidean'))
                    if points_c1.shape[0] > 1: same_distances.extend(pdist(points_c1, metric='euclidean'))
                    if same_distances: 
                        metrics['mean_distance_same'] = float(np.mean(same_distances))
                        metrics['var_dist_same'] = float(np.var(same_distances))

                    # Mean distance to centroid & variance
                    centroid_dists_all: List[float] = []
                    if points_c0.shape[0] > 0 and 'centroid_c0' in locals():
                        dists_c0_to_centroid = np.linalg.norm(points_c0 - centroid_c0, axis=1)
                        if dists_c0_to_centroid.size > 0:
                            metrics['mean_dist_centroid'] = float(np.mean(dists_c0_to_centroid)) # Overwrites if c1 also computed
                            metrics['var_dist_centroid'] = float(np.var(dists_c0_to_centroid))
                            centroid_dists_all.extend(dists_c0_to_centroid)
                    if points_c1.shape[0] > 0 and 'centroid_c1' in locals():
                        dists_c1_to_centroid = np.linalg.norm(points_c1 - centroid_c1, axis=1)
                        if dists_c1_to_centroid.size > 0:
                            # Average the mean distances if both classes present, or handle as needed
                            if metrics['mean_dist_centroid'] is not np.nan :
                                metrics['mean_dist_centroid'] = float(np.mean([metrics['mean_dist_centroid'], np.mean(dists_c1_to_centroid)]))
                                metrics['var_dist_centroid'] = float(np.mean([metrics['var_dist_centroid'], np.var(dists_c1_to_centroid)]))
                            else:
                                metrics['mean_dist_centroid'] = float(np.mean(dists_c1_to_centroid))
                                metrics['var_dist_centroid'] = float(np.var(dists_c1_to_centroid))
                            centroid_dists_all.extend(dists_c1_to_centroid)

                    if centroid_dists_all:
                        metrics['mean_dist_centroid_overall'] = float(np.mean(centroid_dists_all))
                        metrics['var_dist_centroid_overall'] = float(np.var(centroid_dists_all))

                    # Variance of q0 and q1 scores
                    q0_vars_list: List[float] = []
                    if q0_c0.size > 0: q0_vars_list.append(float(np.var(q0_c0)))
                    if q0_c1.size > 0: q0_vars_list.append(float(np.var(q0_c1)))
                    if q0_vars_list: metrics['var_q0'] = float(np.mean(q0_vars_list))
                    
                    q1_vars_list: List[float] = []
                    if q1_c0.size > 0: q1_vars_list.append(float(np.var(q1_c0)))
                    if q1_c1.size > 0: q1_vars_list.append(float(np.var(q1_c1)))
                    if q1_vars_list: metrics['var_q1'] = float(np.mean(q1_vars_list))

                    all_spatial_results[dname][k_val][h_val][model_name] = metrics #type: ignore
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_results_file = output_dir / 'spatial_results.json'
    try:
        # Replace NaN with None for JSON compatibility if needed, or handle them
        # For simplicity, we'll try to save as is. np.nan_to_num might be useful.
        # A more robust way is to recursively convert NaNs to None (or string 'NaN')
        def nan_to_none_or_str(obj):
            if isinstance(obj, dict):
                return {k: nan_to_none_or_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [nan_to_none_or_str(elem) for elem in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None # Or "NaN" as a string
            return obj

        results_to_save = nan_to_none_or_str(all_spatial_results)

        with open(spatial_results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logging.info(f"✅ Likelihood analysis completed! Results saved to {spatial_results_file}")
    except TypeError as e:
        logging.error(f"Could not serialize spatial results to JSON: {e}")
        # Fallback if needed
        try:
            with open(output_dir / 'spatial_results_nonstandard.json', 'w') as f:
                json.dump(all_spatial_results, f, indent=2) # May include NaNs directly
            logging.warning("Saved spatial results to spatial_results_nonstandard.json with NaNs/Infs.")
        except Exception as e_fallback:
            logging.error(f"Fallback saving for spatial results also failed: {e_fallback}")


if __name__ == '__main__':
    start_time = time.time()
    
    # Define base directory for data and output relative to the script file
    # SCRIPT_DIR = Path(__file__).resolve().parent
    # DATA_DIR = SCRIPT_DIR / 'sets'
    # OUTPUT_DIR = SCRIPT_DIR / 'output'
    
    # Using current working directory for simplicity if datasets are there
    DATA_DIR = Path('./sets')
    OUTPUT_DIR = Path('./output')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    loaded_datasets = load_datasets(data_dir=DATA_DIR)

    if not loaded_datasets:
        logging.warning("No datasets were loaded. Exiting analysis.")
    else:
        run_statistical_validation(loaded_datasets, output_dir=OUTPUT_DIR)
        run_likelihood_analysis(loaded_datasets, output_dir=OUTPUT_DIR)
        logging.info(f"\n🎉 Analysis complete! Results saved to {OUTPUT_DIR.resolve()}/ directory")
        
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")