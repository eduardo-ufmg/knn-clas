import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from load_proto import load_test_samples, load_predicted_samples

def main():
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  data_dir = project_root / "data"
  bin_dir = project_root / "bin"
  knn_fit_bin = bin_dir / "knn-fit"
  knn_like_bin = bin_dir / "knn-like"

  # List of dataset names based on real_prepare.py
  datasets = [
    "Breast Cancer",
    "Pima Diabetes",
    "Haberman",
    "Banknote",
    "Sonar",
    "Binary Digits",
    "Ionosphere",
    "SPECT Heart"
  ]

  k = 5  # You can adjust the k value here

  for name in datasets:
    print(f"Processing dataset: {name}")
    fit_path = data_dir / f"{name}_complete_fit.pb"
    test_path = data_dir / f"{name}_complete_pred.pb"
    support_path = data_dir / f"{name}_knn-clas_complete_support.pb"
    likelihoods_path = data_dir / f"{name}_knn-clas_complete_likelihoods_k{k}.pb"

    # Generate support samples using knn-fit
    try:
      subprocess.run(
        [str(knn_fit_bin), str(fit_path), str(support_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
      )
    except subprocess.CalledProcessError as e:
      print(f"Error generating support samples for {name}: {e}")
      continue
    except FileNotFoundError as e:
      print(f"Binary not found: {e}")
      continue

    # Compute likelihoods using knn-like
    try:
      subprocess.run(
        [str(knn_like_bin), str(test_path), str(support_path), str(likelihoods_path), str(k)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
      )
    except subprocess.CalledProcessError as e:
      print(f"Error computing likelihoods for {name}: {e}")
      continue

    # Load predicted likelihoods
    predicted_samples = load_predicted_samples(str(likelihoods_path))
    if not predicted_samples:
      print(f"Failed to load predicted samples for {name}")
      continue

    # Load ground truth labels from test samples
    test_samples = load_test_samples(str(test_path))
    if not test_samples:
      print(f"Failed to load test samples for {name}")
      continue

    # Extract likelihoods and true labels
    likelihood0 = []
    likelihood1 = []
    y_true = []
    target_labels = set()  # Collect unique target labels for axis labeling
    for pred_entry, test_entry in zip(predicted_samples.entries, test_samples.entries):
      # Access likelihoods from the Likelihoods message
      likelihood0.append(pred_entry.likelihoods.likelihood0.likelihood)
      likelihood1.append(pred_entry.likelihoods.likelihood1.likelihood)

      # Access the ground truth target (assuming target_int is used)
      if test_entry.ground_truth.HasField("target_int"):
        y_true.append(test_entry.ground_truth.target_int)
        target_labels.add(test_entry.ground_truth.target_int)
      elif test_entry.ground_truth.HasField("target_str"):
        y_true.append(test_entry.ground_truth.target_str)
        target_labels.add(test_entry.ground_truth.target_str)
      else:
        print(f"Unknown ground truth format for {name}")
        continue

    # Sort target labels for consistent axis labeling
    target_labels = sorted(target_labels)

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Map y_true to integers for coloring
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)

    scatter = plt.scatter(likelihood0, likelihood1, c=y_true_encoded)
    plt.title(f'Likelihood Scatter Plot for {name} (k={k})')
    plt.xlabel(target_labels[0])
    plt.ylabel(target_labels[1])
    plt.grid(True)

    # Save plot
    plot_path = script_dir / f"comparison_results/{name}_likelihood_scatter_k{k}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved scatter plot to {plot_path}")

if __name__ == "__main__":
  main()