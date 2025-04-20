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

def make_grid(X: np.ndarray, target_points: int = 10000) -> np.ndarray:
  """
  Create an adaptive grid of points for decision boundaries, balancing resolution and computation time.
  
  Parameters:
    X (np.ndarray): Input data points.
    target_points (int): Approximate number of grid points (default: 10000).
  
  Returns:
    np.ndarray: Grid of points shaped (n_samples, 2).
  """
  # Calculate data range with padding
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

  # Handle edge cases where range is zero
  x_range = x_max - x_min
  y_range = y_max - y_min
  if x_range <= 0:
    x_range = 1e-5
    x_max = x_min + x_range
  if y_range <= 0:
    y_range = 1e-5
    y_max = y_min + y_range

  # Calculate aspect ratio and determine grid dimensions
  aspect_ratio = x_range / y_range
  nx = int(np.round(np.sqrt(target_points * aspect_ratio)))
  ny = int(np.round(np.sqrt(target_points / aspect_ratio)))
  nx, ny = max(nx, 1), max(ny, 1)  # Ensure at least 1 point per axis

  # Generate evenly spaced grid
  xx = np.linspace(x_min, x_max, nx)
  yy = np.linspace(y_min, y_max, ny)
  xx_mesh, yy_mesh = np.meshgrid(xx, yy)
  
  return np.c_[xx_mesh.ravel(), yy_mesh.ravel()]

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
  