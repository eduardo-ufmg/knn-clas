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
    "--tolerance",
    type=float,
    help="Optional tolerance parameter for the fit step."
  )
  args = parser.parse_args()

  # Construct file paths
  train_path = args.output_dir / "spirals_train.pb"
  test_path = args.output_dir / "spirals_test.pb"
  support_path = args.output_dir / "spirals_support.pb"
  predicted_path = args.output_dir / "spirals_predicted.pb"

  fit_bin = args.bin_dir / "fit"
  pred_bin = args.bin_dir / "pred"

  # Check if binaries exist
  if not fit_bin.exists():
    print(f"Error: fit executable not found at {fit_bin}")
    return
  if not pred_bin.exists():
    print(f"Error: pred executable not found at {pred_bin}")
    return

  # Check input files
  if not train_path.exists():
    print(f"Error: Training data not found at {train_path}")
    return
  if not test_path.exists():
    print(f"Error: Test data not found at {test_path}")
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