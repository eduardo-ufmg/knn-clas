import os
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(X: np.ndarray, y: np.ndarray, name: str) -> None:
  """
  Plot a 2D dataset with points colored by their class.
  Args:
    X (np.ndarray): The input data points.
    y (np.ndarray): The class labels for the data points.
    name (str): The name of the dataset.
  """
  plt.figure()
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
  plt.title(f"Dataset: {name}")
  plt.show()

def plot_test(X: np.ndarray, ytruth: np.ndarray, ypred: np.ndarray, name: str) -> None:
  """
  Plot a test dataset with points colored by their class and outlined by the correctness of the prediction.
  Args:
    X (np.ndarray): The input data points.
    ytruth (np.ndarray): The true class labels for the data points.
    ypred (np.ndarray): The predicted class labels for the data points.
    name (str): The name of the dataset.
  """
  plt.figure()
  correct = ytruth == ypred
  accuracy = np.mean(correct) * 100
  edge_colors = ['green' if c else 'red' for c in correct]
  plt.scatter(X[:, 0], X[:, 1], c=ypred, edgecolor=edge_colors)
  plt.title(f"Test Results for {name} (Accuracy: {accuracy:.2f}%)")
  plt.show()

def plot_decision_boundary(X: np.ndarray, yX: np.ndarray, G: np.ndarray, yG: np.ndarray, name: str) -> None:
  """
  Plot the decision boundary of a classifier.
  Args:
    X (np.ndarray): The input data points.
    yX (np.ndarray): The class labels for the data points.
    G (np.ndarray): The grid points for the decision boundary.
    yG (np.ndarray): The class labels for the grid points.
    name (str): The name of the dataset.
  """
  plt.figure()
  
  unique_x = np.unique(G[:, 0])
  unique_y = np.unique(G[:, 1])
  nx = len(unique_x)
  ny = len(unique_y)
  xx, yy = np.meshgrid(unique_x, unique_y)
  Z = yG.reshape(ny, nx)

  plt.pcolormesh(xx, yy, Z, alpha=0.25)
  plt.scatter(X[:, 0], X[:, 1], c=yX, edgecolor='k')
  plt.title(f"Decision Boundary for {name}")
  plt.xlim(unique_x.min(), unique_x.max())
  plt.ylim(unique_y.min(), unique_y.max())
  plt.show()

if __name__ == "__main__":
  # Example usage
  
  from sklearn.datasets import make_classification
  from generate import make_grid
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split

  # Generate a synthetic dataset
  X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0)

  # Split the dataset into training and testing sets
  X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

  # Train a classifier
  classifier = KNeighborsClassifier()
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X)

  # Create a grid for decision boundary
  G = make_grid(X)
  yG = classifier.predict(G)

  # Plot the dataset
  plot_dataset(X_train, y_train, "Synthetic Dataset")
  plot_test(X, y, y_pred, "Synthetic Dataset Test")
  plot_decision_boundary(X, y, G, yG, "Synthetic Dataset Decision Boundary")