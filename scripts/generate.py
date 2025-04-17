import numpy as np

def make_spirals(n_samples: int=100, noise: float=0.0, turns: int=1) -> tuple[np.ndarray, np.ndarray]:
  """
  Generate a spirals dataset.

  Parameters
  ----------
  n_samples : int, optional
    The number of samples to generate, by default 100
  noise : float, optional
    The standard deviation of the Gaussian noise added to the data, by default 0.0

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
    A tuple containing the generated data points and their corresponding labels.
  """

  samples_per_class = n_samples // 2

  n = np.sqrt(np.random.rand(n_samples)) * 2 * turns * np.pi
  x = n * np.sin(n) + np.random.randn(n_samples) * noise
  y = n * np.cos(n) + np.random.randn(n_samples) * noise
  
  X = np.array(list(zip(x, y)))
  y_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
  
  # Rotate one class 180 degrees
  X[samples_per_class:] = -X[samples_per_class:]
  
  return X, y_labels

def make_grid(X: np.ndarray) -> np.ndarray:
  """
  Create a grid of points for plotting decision boundaries.

  Parameters
  ----------
  X : np.ndarray
    The input data points.

  Returns
  -------
  np.ndarray
    The grid of points.
  """
  
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                       np.arange(y_min, y_max, 1))
  
  return np.c_[xx.ravel(), yy.ravel()]

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  
  X, y = make_spirals(1000, 0.1, 2)

  from sklearn.neighbors import KNeighborsClassifier

  knn = KNeighborsClassifier()

  knn.fit(X, y)
  grid = make_grid(X)
  grid_predictions = knn.predict(grid)

  plt.scatter(X[:, 0], X[:, 1], c=y)
  plt.scatter(grid[:, 0], grid[:, 1], c=grid_predictions, marker='s', alpha=0.25)
  plt.show()
  