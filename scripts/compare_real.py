import subprocess
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_proto import load_test_samples, load_predicted_samples

def main():
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  data_dir = project_root / "data"
  bin_dir = project_root / "bin"

  # Discover datasets by checking for test files
  test_files = list(data_dir.glob("*_test.pb"))
  datasets = []
  for test_file in test_files:
    dataset_name = test_file.stem.split("_test")[0]
    train_file = data_dir / f"{dataset_name}_train.pb"
    if train_file.exists():
      datasets.append(dataset_name)

  if not datasets:
    print("No datasets found.")
    return

  results = []
  nn_tolerance = 0.0
  knn_ks = [1, 3, 5]

  for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    train_path = data_dir / f"{dataset}_train.pb"
    test_path = data_dir / f"{dataset}_test.pb"

    # Load ground truth
    test_samples = load_test_samples(str(test_path))
    if test_samples is None:
      print(f"Failed to load test samples for {dataset}")
      continue
    y_true = [entry.ground_truth.target_int for entry in test_samples.entries]

    # Run nn-clas model
    nn_fit_bin = bin_dir / "nn-fit"
    nn_pred_bin = bin_dir / "nn-pred"
    if nn_fit_bin.exists() and nn_pred_bin.exists():
      nn_support_path = data_dir / f"{dataset}_nn-clas_support.pb"
      nn_predicted_path = data_dir / f"{dataset}_nn-clas_predicted.pb"

      try:
        # Fit
        subprocess.run([
          str(nn_fit_bin), str(train_path),
          str(nn_support_path), str(nn_tolerance)
        ], check=True)
        # Predict
        subprocess.run([
          str(nn_pred_bin), str(test_path),
          str(nn_support_path), str(nn_predicted_path)
        ], check=True)

        # Load predictions
        nn_predicted = load_predicted_samples(str(nn_predicted_path))
        if nn_predicted:
          y_pred = [entry.target.target_int for entry in nn_predicted.entries]
          accuracy = accuracy_score(y_true, y_pred)
          precision = precision_score(y_true, y_pred, average='macro')
          recall = recall_score(y_true, y_pred, average='macro')
          f1 = f1_score(y_true, y_pred, average='macro')
          results.append({
            'dataset': dataset,
            'model': 'nn-clas',
            'k': None,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
          })
      except subprocess.CalledProcessError as e:
        print(f"nn-clas failed for {dataset}: {e}")

    # Run knn-clas for each k
    knn_fit_bin = bin_dir / "knn-fit"
    knn_pred_bin = bin_dir / "knn-pred"
    if knn_fit_bin.exists() and knn_pred_bin.exists():
      knn_support_path = data_dir / f"{dataset}_knn-clas_support.pb"
      try:
        # Fit once
        subprocess.run([
          str(knn_fit_bin), str(train_path), str(knn_support_path)
        ], check=True)
      except subprocess.CalledProcessError as e:
        print(f"knn-fit failed for {dataset}: {e}")
        continue

      for k in knn_ks:
        knn_predicted_path = data_dir / f"{dataset}_knn-clas_predicted_k{k}.pb"
        try:
          # Predict with current k
          subprocess.run([
            str(knn_pred_bin), str(test_path),
            str(knn_support_path), str(knn_predicted_path), str(k)
          ], check=True)

          # Load predictions
          knn_predicted = load_predicted_samples(str(knn_predicted_path))
          if knn_predicted:
            y_pred = [entry.target.target_int for entry in knn_predicted.entries]
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            results.append({
              'dataset': dataset,
              'model': 'knn-clas',
              'k': k,
              'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'f1': f1
            })
        except subprocess.CalledProcessError as e:
          print(f"knn-pred (k={k}) failed for {dataset}: {e}")

  # Print results
  print("\nComparison Results:")
  print("{:<20} {:<10} {:<5} {:<8} {:<8} {:<8} {:<8}".format(
    "Dataset", "Model", "k", "Acc", "Prec", "Rec", "F1"))
  for res in results:
    k = res['k'] if res['k'] is not None else ''
    print("{:<20} {:<10} {:<5} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f}".format(
      res['dataset'], res['model'], k,
      res['accuracy'], res['precision'], res['recall'], res['f1']
    ))

if __name__ == "__main__":
  main()
  