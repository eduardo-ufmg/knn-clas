import subprocess
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_proto import load_test_samples, load_predicted_samples

def main():
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  data_dir = project_root / "data"
  bin_dir = project_root / "bin"
  nn_fit_bin = bin_dir / "nn-fit"
  nn_pred_bin = bin_dir / "nn-pred"
  knn_fit_bin = bin_dir / "knn-fit"
  knn_pred_bin = bin_dir / "knn-pred"

  # Load metadata
  metadata_path = script_dir / "comparison_results" / "setsmetadata.csv"
  metadata_dict = {}
  try:
    with open(metadata_path, 'r') as f:
      reader = csv.DictReader(f)
      for row in reader:
        metadata_dict[row['name']] = {
          'nsamples': row['nsamples'],
          'nfeatures': row['nfeatures']
        }
  except FileNotFoundError:
    print(f"Metadata file {metadata_path} not found.")
    return

  # Discover datasets and folds
  datasets = defaultdict(list)
  test_files = list(data_dir.glob("*_test.pb"))
  for test_file in test_files:
    stem = test_file.stem
    if "_fold" in stem:
      parts = stem.split("_fold")
      base_name = parts[0]
      fold_part = parts[1].split("_test")[0]
      try:
        fold = int(fold_part)
        datasets[base_name].append(fold)
      except ValueError:
        continue
    else:
      base_name = stem.split("_test")[0]
      datasets[base_name].append(0)

  if not datasets:
    print("No datasets found.")
    return

  nn_tolerance = 0.0
  knn_ks = [1, 3, 5]

  results = []
  for dataset in datasets:
    folds = sorted(datasets[dataset])
    print(f"Processing {dataset} with {len(folds)} folds")

    nn_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    knn_metrics = {k: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for k in knn_ks}

    for fold in folds:
      train_path = data_dir / f"{dataset}_fold{fold}_train.pb"
      test_path = data_dir / f"{dataset}_fold{fold}_test.pb"

      test_samples = load_test_samples(str(test_path))
      if not test_samples:
        continue
      y_true = [entry.ground_truth.target_int for entry in test_samples.entries]

      # Process nn-clas
      nn_support_path = data_dir / f"{dataset}_nn-clas_fold{fold}_support.pb"
      nn_predicted_path = data_dir / f"{dataset}_nn-clas_fold{fold}_predicted.pb"
      try:
        subprocess.run([str(nn_fit_bin), str(train_path), str(nn_support_path), str(nn_tolerance)], check=True)
        subprocess.run([str(nn_pred_bin), str(test_path), str(nn_support_path), str(nn_predicted_path)], check=True)
        nn_predicted = load_predicted_samples(str(nn_predicted_path))
        if nn_predicted:
          y_pred = [entry.target.target_int for entry in nn_predicted.entries]
          nn_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
          nn_metrics['precision'].append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
          nn_metrics['recall'].append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
          nn_metrics['f1'].append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
      except subprocess.CalledProcessError as e:
        print(f"nn-clas fold {fold} failed: {e}")

      # Process knn-clas
      knn_support_path = data_dir / f"{dataset}_knn-clas_fold{fold}_support.pb"
      try:
        subprocess.run([str(knn_fit_bin), str(train_path), str(knn_support_path)], check=True)
        for k in knn_ks:
          knn_predicted_path = data_dir / f"{dataset}_knn-clas_fold{fold}_predicted_k{k}.pb"
          subprocess.run([str(knn_pred_bin), str(test_path), str(knn_support_path), str(knn_predicted_path), str(k)], check=True)
          knn_predicted = load_predicted_samples(str(knn_predicted_path))
          if knn_predicted:
            y_pred = [entry.target.target_int for entry in knn_predicted.entries]
            knn_metrics[k]['accuracy'].append(accuracy_score(y_true, y_pred))
            knn_metrics[k]['precision'].append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            knn_metrics[k]['recall'].append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            knn_metrics[k]['f1'].append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
      except subprocess.CalledProcessError as e:
        print(f"knn-clas fold {fold} failed: {e}")

    # Compute averages for nn-clas
    if nn_metrics['accuracy']:
      dataset_meta = metadata_dict.get(dataset, {})
      results.append({
        'Dataset': dataset,
        'nSamples': dataset_meta.get('nsamples', ''),
        'nFeatures': dataset_meta.get('nfeatures', ''),
        'Model': 'nn-clas',
        'k': '',
        'Accuracy': f"{np.mean(nn_metrics['accuracy']):.2f} ± {np.std(nn_metrics['accuracy']):.2f}",
        'Precision': f"{np.mean(nn_metrics['precision']):.2f} ± {np.std(nn_metrics['precision']):.2f}",
        'Recall': f"{np.mean(nn_metrics['recall']):.2f} ± {np.std(nn_metrics['recall']):.2f}",
        'F1': f"{np.mean(nn_metrics['f1']):.2f} ± {np.std(nn_metrics['f1']):.2f}",
      })

    # Compute averages for knn-clas
    for k in knn_ks:
      if knn_metrics[k]['accuracy']:
        dataset_meta = metadata_dict.get(dataset, {})
        results.append({
          'Dataset': dataset,
          'nSamples': dataset_meta.get('nsamples', ''),
          'nFeatures': dataset_meta.get('nfeatures', ''),
          'Model': 'knn-clas',
          'k': k,
          'Accuracy': f"{np.mean(knn_metrics[k]['accuracy']):.2f} ± {np.std(knn_metrics[k]['accuracy']):.2f}",
          'Precision': f"{np.mean(knn_metrics[k]['precision']):.2f} ± {np.std(knn_metrics[k]['precision']):.2f}",
          'Recall': f"{np.mean(knn_metrics[k]['recall']):.2f} ± {np.std(knn_metrics[k]['recall']):.2f}",
          'F1': f"{np.mean(knn_metrics[k]['f1']):.2f} ± {np.std(knn_metrics[k]['f1']):.2f}",
        })

  # Print results
  print("\nComparison Results:")
  print("{:<20} {:<10} {:<10} {:<10} {:<5} {:<20} {:<20} {:<20} {:<20}".format(
    "Dataset", "nSamples", "nFeatures", "Model", "k", "Accuracy", "Precision", "Recall", "F1"))
  for res in results:
    print("{:<20} {:<10} {:<10} {:<10} {:<5} {:<20} {:<20} {:<20} {:<20}".format(
      res['Dataset'], res['nSamples'], res['nFeatures'], res['Model'], res['k'],
      res['Accuracy'], res['Precision'], res['Recall'], res['F1']
    ))

  # Write results to CSV
  output_dir = script_dir / "comparison_results"
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / "real_sets.csv"
  with open(output_file, "w") as f:
    f.write("Dataset,nSamples,nFeatures,Model,k,Accuracy,Precision,Recall,F1\n")
    for res in results:
      line = (
        f"{res['Dataset']},"
        f"{res['nSamples']},"
        f"{res['nFeatures']},"
        f"{res['Model']},"
        f"{res['k']},"
        f"\"{res['Accuracy']}\","
        f"\"{res['Precision']}\","
        f"\"{res['Recall']}\","
        f"\"{res['F1']}\"\n"
      )
      f.write(line)

if __name__ == "__main__":
  main()