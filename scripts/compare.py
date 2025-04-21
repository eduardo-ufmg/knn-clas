import os
import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure the script can import modules from the scripts directory
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from load_proto import (
  load_test_samples,
  load_predicted_samples,
  load_support_samples,
  load_dataset
)

def run_model(model, output_dir, bin_dir, k=None, tolerance=None):
  """Run the specified model using subprocess with optional parameters."""
  run_script = script_dir / "run.py"
  cmd = [
    sys.executable,
    str(run_script),
    "--model", model,
    "--output_dir", str(output_dir),
    "--bin_dir", str(bin_dir)
  ]

  # Add tolerance for nn-clas
  if model == "nn-clas" and tolerance is not None:
    cmd.extend(["--tolerance", str(tolerance)])
  # Add k for knn-clas
  elif model == "knn-clas" and k is not None:
    cmd.extend(["--k", str(k)])

  try:
    subprocess.run(cmd, check=True)
    print(f"Successfully ran {model} model.")
  except subprocess.CalledProcessError as e:
    print(f"Error running model {model}: {e}")
    sys.exit(1)

def main():
  parser = argparse.ArgumentParser(description="Compare models with optional parameters.")
  parser.add_argument('--k', type=int, default=2, help='Number of neighbors for knn-clas model')
  parser.add_argument('--tolerance', type=float, default=0.0, help='Tolerance parameter for nn-clas model')
  args = parser.parse_args()

  project_root = script_dir.parent
  output_dir = project_root / "data"
  bin_dir = project_root / "bin"

  # Get arguments
  k = args.k
  tolerance = args.tolerance

  # Run both models with respective parameters
  for model in ["nn-clas", "knn-clas"]:
    if model == "nn-clas":
      run_model(model, output_dir, bin_dir, tolerance=tolerance)
    else:
      run_model(model, output_dir, bin_dir, k=k)

  # Load test samples
  test_path = output_dir / "spirals_test.pb"
  test_samples = load_test_samples(str(test_path))
  if test_samples is None:
    print("Error: Failed to load test samples.")
    return

  # Load training data
  train_path = output_dir / "spirals_train.pb"
  train_data = load_dataset(str(train_path))
  if train_data is None:
    print("Error: Failed to load training data.")
    return

  # Prepare data for plotting
  X_test = np.array([[entry.features[0], entry.features[1]] for entry in test_samples.entries])
  y_truth = np.array([entry.ground_truth.target_int for entry in test_samples.entries])
  X_train = np.array([[entry.features[0], entry.features[1]] for entry in train_data.entries])
  y_train = np.array([entry.target.target_int for entry in train_data.entries])

  models = ["nn-clas", "knn-clas"]
  accuracies = {}
  support_data = {}
  predicted_data = {}

  # Load results for each model
  for model in models:
    # Determine the correct file prefixes
    prefix = "nn" if model == "nn-clas" else "knn"
    predicted_path = output_dir / f"spirals_{prefix}_predicted.pb"
    support_path = output_dir / f"spirals_{prefix}_support.pb"

    # Load predicted and support samples
    predicted = load_predicted_samples(str(predicted_path))
    if predicted is None:
      print(f"Error: Failed to load predicted samples for {model}.")
      return
    support = load_support_samples(str(support_path))
    if support is None:
      print(f"Error: Failed to load support samples for {model}.")
      return

    # Calculate accuracy
    y_pred = np.array([entry.target.target_int for entry in predicted.entries])
    accuracy = np.mean(y_truth == y_pred) * 100
    accuracies[model] = accuracy
    predicted_data[model] = (X_test, y_pred)
    support_data[model] = (support, y_truth)

  # Create comparison plot
  plt.figure(figsize=(12, 6))

  for idx, model in enumerate(models, 1):
    plt.subplot(1, 2, idx)
    support, _ = support_data[model]
    X_support = np.array([[entry.features[0], entry.features[1]] for entry in support.entries])
    y_support = np.array([entry.target.target_int for entry in support.entries])

    X_test_plot, y_pred = predicted_data[model]
    correct = y_truth == y_pred
    edge_colors = ['green' if c else 'red' for c in correct]

    # Plot test points with prediction correctness
    plt.scatter(
      X_test_plot[:, 0], X_test_plot[:, 1],
      c=y_pred,
      edgecolors=edge_colors,
    )

    # Plot support vectors
    plt.scatter(
      X_support[:, 0], X_support[:, 1],
      c=y_support,
      marker='x',
    )

    # Plot training data
    plt.scatter(
      X_train[:, 0], X_train[:, 1],
      c=y_train,
      edgecolors='black',
      alpha=0.3,
    )

    if model == "nn-clas":
      title = f"{model}/tol={tolerance:.2f}"
    else:
      title = f"{model}/k={k}" 

    plt.title(f"{title} - Accuracy: {accuracies[model]:.1f}%")

  plt.tight_layout()

  # Ensure the comparison_results directory exists
  comparison_results_dir = script_dir / "comparison_results"
  comparison_results_dir.mkdir(exist_ok=True)

  # Save the plot to the comparison_results directory
  plot_path = comparison_results_dir / f"nn_{k}nn.png"
  plt.savefig(plot_path)
  print(f"Comparison plot saved to {plot_path}")

  plt.show()

if __name__ == "__main__":
  main()
