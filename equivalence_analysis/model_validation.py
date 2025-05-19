import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from equivalence_analysis import KNN, KNN_CLAS

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0)

y = np.where(y == 0, -1, 1)  # Convert to -1 and 1 for KNN

knn = KNN().fit(X, y)

knn_clas = KNN_CLAS().fit(X, y)

knn_q0, knn_q1 = knn.likelihood_score(X)
knn_clas_q0, knn_clas_q1 = knn_clas.likelihood_score(X)

y = np.where(y == -1, 0, 1)  # Convert back to 0 and 1 for plotting

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(knn_q0, knn_q1, alpha=0.5, c=y, label='Samples')
axes[0].set_title('KNN Likelihood Space')
axes[0].set_xlabel('q0')
axes[0].set_ylabel('q1')
axes[0].legend()

axes[1].scatter(knn_clas_q0, knn_clas_q1, alpha=0.5, c=y, label='Samples')
axes[1].set_title('KNN_CLAS Likelihood Space')
axes[1].set_xlabel('q0')
axes[1].set_ylabel('q1')
axes[1].legend()

plt.tight_layout()
plt.show()
