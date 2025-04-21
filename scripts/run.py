import argparse
import subprocess
import os
from pathlib import Path

def main():
  parser = argparse.ArgumentParser(description="Run the fit and prediction steps.")
  parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("data"),
    help="Directory containing input data and where output files will be saved."
  )
  parser.add_argument(
    "--bin_dir",
    type=Path,
    default=Path("bin"),
    help="Directory containing fit and pred executables."
  )
  parser.add_argument(
    "--model",
    type=str,
    choices=["nn-clas", "knn-clas"],
    default="nn-clas",
    help="Model type to use for fitting and prediction."
  )
  parser.add_argument(
    "--tolerance",
    type=float,
    help="Optional tolerance parameter for the fit step."
  )
  parser.add_argument(
    "--k",
    type=int,
    default=2,
    help="Number of neighbors for the knn-clas model."
  )
  args = parser.parse_args()

  # Construct file paths
  train_path = args.output_dir / "spirals_train.pb"
  test_path = args.output_dir / "spirals_test.pb"
  
  # Support and prediction paths for the nn-clas model
  nn_support_path = args.output_dir / "spirals_nn_support.pb"
  nn_predicted_path = args.output_dir / "spirals_nn_predicted.pb"

  # Support and prediction paths for the knn-clas model
  knn_support_path = args.output_dir / "spirals_knn_support.pb"
  knn_predicted_path = args.output_dir / "spirals_knn_predicted.pb"

  # Binaries for nn-clas model
  nn_fit_bin = args.bin_dir / "nn-fit"
  nn_pred_bin = args.bin_dir / "nn-pred"

  # Binaries for knn-clas model
  knn_fit_bin = args.bin_dir / "knn-fit"
  knn_pred_bin = args.bin_dir / "knn-pred"

  # Set model
  model = args.model

  # Check if binaries exist
  if not nn_fit_bin.exists():
    print(f"Error: fit executable not found at {nn_fit_bin}")
    return
  if not nn_pred_bin.exists():
    print(f"Error: pred executable not found at {nn_pred_bin}")
    return
  if not knn_fit_bin.exists():
    print(f"Error: knn-fit executable not found at {knn_fit_bin}")
    return
  if not knn_pred_bin.exists():
    print(f"Error: knn-pred executable not found at {knn_pred_bin}")
    return

  # Check input files
  if not train_path.exists():
    print(f"Error: Training data not found at {train_path}")
    return
  if not test_path.exists():
    print(f"Error: Test data not found at {test_path}")
    return
  
  # Select paths based on model
  if model == "nn-clas":
    support_path = nn_support_path
    predicted_path = nn_predicted_path
    fit_bin = nn_fit_bin
    pred_bin = nn_pred_bin
  elif model == "knn-clas":
    support_path = knn_support_path
    predicted_path = knn_predicted_path
    fit_bin = knn_fit_bin
    pred_bin = knn_pred_bin
  else:
    print(f"Error: Unsupported model type {model}")
    return

  # Run fit step
  fit_cmd = [str(fit_bin), str(train_path), str(support_path)]
  if args.tolerance is not None:
    fit_cmd.append(str(args.tolerance))
  fit_result = subprocess.run(fit_cmd)
  if fit_result.returncode != 0:
    print("Fit step failed with exit code", fit_result.returncode)
    return

  # Run prediction step
  pred_cmd = [str(pred_bin), str(test_path), str(support_path), str(predicted_path)]
  pred_result = subprocess.run(pred_cmd)
  if pred_result.returncode != 0:
    print("Prediction step failed with exit code", pred_result.returncode)
    return

  print("Successfully completed fit and prediction steps.")
  print(f"Support samples saved to {support_path}")
  print(f"Predicted samples saved to {predicted_path}")

if __name__ == "__main__":
  main()