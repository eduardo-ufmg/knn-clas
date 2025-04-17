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
  NotImplementedError("plot_dataset function is not implemented yet.")

def plot_test(X: np.ndarray, ytruth: np.ndarray, ypred: np.ndarray, name: str) -> None:
  """
  Plot a test dataset with points colored by their class and outlined by the correctness of the prediction.
  Args:
    X (np.ndarray): The input data points.
    ytruth (np.ndarray): The true class labels for the data points.
    ypred (np.ndarray): The predicted class labels for the data points.
    name (str): The name of the dataset.
  """
  NotImplementedError("plot_test function is not implemented yet.")

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
  NotImplementedError("plot_decision_boundary function is not implemented yet.")