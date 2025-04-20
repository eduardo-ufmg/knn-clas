import os
import pathlib
import argparse

from sklearn.model_selection import train_test_split

from classifier_pb2 import Dataset, TestSamples

from generate import make_spirals
from store_proto import store_dataset, store_test_samples

def main():
  parser = argparse.ArgumentParser(description="Prepare the dataset.")
  parser.add_argument(
    "--output_dir",
    type=pathlib.Path,
    default=pathlib.Path("data") / "input",
    help="Directory to store the dataset.",
  )
  parser.add_argument(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of samples to generate.",
  )
  parser.add_argument(
    "--noise",
    type=float,
    default=0.1,
    help="Standard deviation of the Gaussian noise.",
  )
  parser.add_argument(
    "--test_ratio",
    type=float,
    default=0.2,
    help="Ratio of test samples to total samples.",
  )

  args = parser.parse_args()
  output_dir = args.output_dir
  num_samples = args.num_samples
  noise = args.noise
  test_ratio = args.test_ratio

  # Create the output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Generate the dataset
  X, y = make_spirals(num_samples, noise, turns=2)

  # Split the dataset into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio
  )

  # Create the training dataset
  train_dataset = Dataset()
  for features, target in zip(X_train, y_train):
    entry = train_dataset.entries.add()
    entry.features.extend(features)
    entry.target.target_int = int(target)

  # Store the training dataset
  train_dataset_path = output_dir / "spirals.pb"

  if not store_dataset(train_dataset, str(train_dataset_path)):
    print(f"Failed to store the dataset at {train_dataset_path}")
    return
  
  print(f"Dataset stored at {train_dataset_path}")

  # Create the test samples
  test_samples = TestSamples()
  for features, target in zip(X_test, y_test):
    entry = test_samples.entries.add()
    entry.sample_id = len(test_samples.entries)  # Assign a unique sample ID
    entry.features.extend(features)
    entry.ground_truth.target_int = int(target)

  # Store the test samples
  test_samples_path = output_dir / "spirals_test.pb"
  if not store_test_samples(test_samples, str(test_samples_path)):
    print(f"Failed to store the test samples at {test_samples_path}")
    return
  
  print(f"Test samples stored at {test_samples_path}")

if __name__ == "__main__":
  main()
