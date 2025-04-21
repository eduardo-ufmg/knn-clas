import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from load_proto import load_test_samples, load_predicted_samples, load_support_samples, load_dataset
from plot import plot_test

def main():
  parser = argparse.ArgumentParser(description="Validate and visualize classifier predictions with support vectors.")
  parser.add_argument(
  "--input_dir",
  type=pathlib.Path,
  default=pathlib.Path("data"),
  help="Directory containing test, predicted, and support samples."
  )
  args = parser.parse_args()
  input_dir = args.input_dir

  # Define file paths
  test_path = input_dir / "spirals_test.pb"
  predicted_path = input_dir / "spirals_predicted.pb"
  support_path = input_dir / "spirals_support.pb"
  train_path = input_dir / "spirals_train.pb"

  # Load test samples
  test_samples = load_test_samples(str(test_path))
  if test_samples is None:
    print("Error: Failed to load test samples.")
    return

  # Load predicted samples
  predicted_samples = load_predicted_samples(str(predicted_path))
  if predicted_samples is None:
    print("Error: Failed to load predicted samples.")
    return

  # Load support samples
  support_samples = load_support_samples(str(support_path))
  if support_samples is None:
    print("Error: Failed to load support samples.")
    return
  
  # Load training dataset
  train_data = load_dataset(str(train_path))
  if train_data is None:
    print("Error: Failed to load training dataset.")
    return

  # Extract test data
  X_test = np.array([[entry.features[0], entry.features[1]] for entry in test_samples.entries])
  y_truth = np.array([entry.ground_truth.target_int for entry in test_samples.entries])
  y_pred = np.array([entry.target.target_int for entry in predicted_samples.entries])

  # Extract training data
  X_train = np.array([[entry.features[0], entry.features[1]] for entry in train_data.entries])
  y_train = np.array([entry.target.target_int for entry in train_data.entries])

  # Validate matching sample counts
  if len(y_truth) != len(y_pred):
    print("Error: Mismatch between test and predicted sample counts.")
    return

  # Plot basic test results using existing function
  plot_test(X_test, y_truth, y_pred, "Spirals Dataset")

  # Extract support vectors
  X_support = np.array([[entry.features[0], entry.features[1]] for entry in support_samples.entries])
  y_support = np.array([entry.target.target_int for entry in support_samples.entries])

  # Create enhanced visualization with support vectors
  plt.figure()
  correct = y_truth == y_pred
  accuracy = np.mean(correct) * 100
  edge_colors = ['green' if c else 'red' for c in correct]

  # Plot test points with prediction colors and correctness edges
  scatter = plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_pred,
    edgecolor=edge_colors,
  )

  # Overlay support vectors with true class colors
  plt.scatter(
    X_support[:, 0], X_support[:, 1],
    c=y_support,
    marker='x'
  )

  # Overlay training points with true class colors
  plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train,
    marker='o',
    edgecolor='black',
    alpha=0.5
  )

  plt.title(f"Classifier Predictions with Support Vectors\nAccuracy: {accuracy:.1f}%")
  plt.show()

if __name__ == "__main__":
  main()
