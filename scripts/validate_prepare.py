import numpy as np
from load_proto import load_dataset, load_test_samples
from plot import plot_dataset

def main():
  # Load the training dataset
  train_dataset_path = "data/input/train_dataset.pb"
  train_data = load_dataset(train_dataset_path)
  if train_data is None:
    print("Error: Failed to load training dataset.")
    return
  
  # Extract features and labels from training data
  X_train = np.array([(entry.features[0], entry.features[1]) for entry in train_data.entries])
  y_train = np.array([entry.target.target_int for entry in train_data.entries])
  
  # Plot the training dataset
  plot_dataset(X_train, y_train, "Training Dataset")
  
  # Load the test samples
  test_samples_path = "data/input/test_samples.pb"
  test_samples = load_test_samples(test_samples_path)
  if test_samples is None:
    print("Error: Failed to load test samples.")
    return
  
  # Extract features and labels from test samples
  X_test = np.array([(entry.features[0], entry.features[1]) for entry in test_samples.entries])
  y_test = np.array([entry.ground_truth.target_int for entry in test_samples.entries])
  
  # Plot the test samples
  plot_dataset(X_test, y_test, "Test Samples")

if __name__ == "__main__":
  main()