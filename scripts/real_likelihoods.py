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

  k_values = [1, 3, 5]

  for name in datasets:
    print(f"Processing dataset: {name}")
    fit_path = data_dir / f"{name}_complete_fit.pb"
    test_path = data_dir / f"{name}_complete_pred.pb"
    support_path = data_dir / f"{name}_knn-clas_complete_support.pb"

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

    # Load ground truth labels from test samples
    test_samples = load_test_samples(str(test_path))
    if not test_samples:
      print(f"Failed to load test samples for {name}")
      continue

    # Extract y_true and target labels
    y_true = []
    target_labels = set()
    for test_entry in test_samples.entries:
      if test_entry.ground_truth.HasField("target_int"):
        label = test_entry.ground_truth.target_int
      elif test_entry.ground_truth.HasField("target_str"):
        label = test_entry.ground_truth.target_str
      else:
        print(f"Unknown ground truth format for {name}")
        y_true = None
        break
      y_true.append(label)
      target_labels.add(label)
    
    if y_true is None:
      continue  # Skip dataset if ground truth extraction failed

    target_labels = sorted(target_labels)
    if len(target_labels) != 2:
      print(f"Dataset {name} does not have exactly two target labels")
      continue

    # Encode y_true to integers
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)

    # Process each k value
    all_likelihoods = []
    for k in k_values:
      likelihoods_path = data_dir / f"{name}_knn-clas_complete_likelihoods_k{k}.pb"
      
      # Compute likelihoods using knn-like
      try:
        subprocess.run(
          [str(knn_like_bin), str(test_path), str(support_path), str(likelihoods_path), str(k)],
          check=True,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL
        )
      except subprocess.CalledProcessError as e:
        print(f"Error computing likelihoods for {name} k={k}: {e}")
        all_likelihoods = None
        break
      except FileNotFoundError as e:
        print(f"Binary not found: {e}")
        all_likelihoods = None
        break

      # Load predicted likelihoods
      predicted_samples = load_predicted_samples(str(likelihoods_path))
      if not predicted_samples:
        print(f"Failed to load predicted samples for {name} k={k}")
        all_likelihoods = None
        break

      # Extract likelihoods
      likelihood0 = []
      likelihood1 = []
      for pred_entry in predicted_samples.entries:
        likelihood0.append(pred_entry.likelihoods.likelihood0.likelihood)
        likelihood1.append(pred_entry.likelihoods.likelihood1.likelihood)
      
      all_likelihoods.append((k, likelihood0, likelihood1))

    if all_likelihoods is None or len(all_likelihoods) != 3:
      print(f"Skipping {name} due to errors in processing some k values")
      continue

    # Create a figure with three subplots
    plt.figure(figsize=(20, 6))
    plt.suptitle(f'Likelihood Scatter Plots for {name}', fontsize=14)

    for idx, (k, l0, l1) in enumerate(all_likelihoods, 1):
      plt.subplot(1, 3, idx)
      plt.scatter(l0, l1, c=y_true_encoded, alpha=0.6)
      plt.title(f'k = {k}')
      plt.xlabel(target_labels[0])
      plt.ylabel(target_labels[1])
      plt.grid(True)

    plt.tight_layout()

    # Save plot
    plot_dir = script_dir / "comparison_results"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f"{name}_likelihood.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved scatter plot to {plot_path}")

if __name__ == "__main__":
  main()