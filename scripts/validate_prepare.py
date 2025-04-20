import os
import pathlib
import argparse
import numpy as np

from load_proto import load_dataset, load_test_samples
from plot import plot_dataset

def main():
  parser = argparse.ArgumentParser(description="Validate and visualize datasets.")
  parser.add_argument(
    "--input_dir",
    type=pathlib.Path,
    default=pathlib.Path("data") / "input",
    help="Directory containing the dataset files.",
  )
  args = parser.parse_args()
  input_dir = args.input_dir

  # Paths to dataset files
  train_dataset_path = input_dir / "spirals.pb"
  test_samples_path = input_dir / "spirals_test.pb"

  # Validate and plot the training dataset
  if not os.path.exists(train_dataset_path):
    print(f"Error: Training dataset not found at {train_dataset_path}")
    return

  train_data = load_dataset(str(train_dataset_path))
  if train_data is None:
    print("Error: Failed to load training dataset.")
    return

  X_train = np.array([(entry.features[0], entry.features[1]) for entry in train_data.entries])
  y_train = np.array([entry.target.target_int for entry in train_data.entries])
  plot_dataset(X_train, y_train, "Training Dataset")

  # Validate and plot the test samples
  if not os.path.exists(test_samples_path):
    print(f"Error: Test samples not found at {test_samples_path}")
    return

  test_samples = load_test_samples(str(test_samples_path))
  if test_samples is None:
    print("Error: Failed to load test samples.")
    return

  X_test = np.array([(entry.features[0], entry.features[1]) for entry in test_samples.entries])
  y_test = np.array([entry.ground_truth.target_int for entry in test_samples.entries])
  plot_dataset(X_test, y_test, "Test Samples")

if __name__ == "__main__":
  main()