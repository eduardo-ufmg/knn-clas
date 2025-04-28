import subprocess
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score
from load_proto import load_test_samples, load_predicted_samples, load_support_samples

def main():
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  data_dir = project_root / "data"
  bin_dir = project_root / "bin"
  nn_fit_bin = bin_dir / "nn-fit"
  nn_pred_bin = bin_dir / "nn-pred"
  knn_fit_bin = bin_dir / "knn-fit"
  knn_pred_bin = bin_dir / "knn-pred"
  run_and_measure_script = script_dir / "run_and_measure_time.sh"

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

    nn_metrics = {'accuracy': [], 'train_time': [], 'pred_time': [], 'support_count': []}
    knn_metrics = {
      'train_time': [],
      'support_count': [],
      'k_metrics': {k: {'accuracy': [], 'pred_time': []} for k in knn_ks}
    }

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
        # Measure training time
        train_time_output = subprocess.check_output(
          [str(run_and_measure_script), str(nn_fit_bin), str(train_path), str(nn_support_path), str(nn_tolerance)],
          text=True
        )
        train_time_ms = float(train_time_output.strip())
        nn_metrics['train_time'].append(train_time_ms)

        # Load and count support samples
        support_samples = load_support_samples(str(nn_support_path))
        if support_samples:
          nn_metrics['support_count'].append(len(support_samples.entries))

        # Measure prediction time
        pred_time_output = subprocess.check_output(
          [str(run_and_measure_script), str(nn_pred_bin), str(test_path), str(nn_support_path), str(nn_predicted_path)],
          text=True
        )
        pred_time_ms = float(pred_time_output.strip())
        nn_metrics['pred_time'].append(pred_time_ms)

        nn_predicted = load_predicted_samples(str(nn_predicted_path))
        if nn_predicted:
          y_pred = [
            entry.target.target_str if entry.target.target_str else entry.target.target_int
            for entry in nn_predicted.entries
          ]
          y_true = [
            entry.ground_truth.target_str if entry.ground_truth.target_str else entry.ground_truth.target_int
            for entry in test_samples.entries
          ]
          nn_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
      except subprocess.CalledProcessError as e:
        print(f"nn-clas fold {fold} failed: {e}")

      # Process knn-clas
      knn_support_path = data_dir / f"{dataset}_knn-clas_fold{fold}_support.pb"
      try:
        # Measure training time
        train_time_output = subprocess.check_output(
          [str(run_and_measure_script), str(knn_fit_bin), str(train_path), str(knn_support_path)],
          text=True
        )
        train_time_ms = float(train_time_output.strip())
        knn_metrics['train_time'].append(train_time_ms)

        # Load and count support samples
        support_samples = load_support_samples(str(knn_support_path))
        if support_samples:
          knn_metrics['support_count'].append(len(support_samples.entries))

        for k in knn_ks:
          knn_predicted_path = data_dir / f"{dataset}_knn-clas_fold{fold}_predicted_k{k}.pb"
          # Measure prediction time for k
          pred_time_output = subprocess.check_output(
            [str(run_and_measure_script), str(knn_pred_bin), str(test_path), str(knn_support_path), str(knn_predicted_path), str(k)],
            text=True
          )
          pred_time_ms = float(pred_time_output.strip())
          knn_metrics['k_metrics'][k]['pred_time'].append(pred_time_ms)

          knn_predicted = load_predicted_samples(str(knn_predicted_path))
          if knn_predicted:
            y_pred = [
              entry.target.target_str if entry.target.target_str else entry.target.target_int
              for entry in knn_predicted.entries
            ]
            y_true = [
              entry.ground_truth.target_str if entry.ground_truth.target_str else entry.ground_truth.target_int
              for entry in test_samples.entries
            ]
            knn_metrics['k_metrics'][k]['accuracy'].append(accuracy_score(y_true, y_pred))
      except subprocess.CalledProcessError as e:
        print(f"knn-clas fold {fold} failed: {e}")

    # Compute averages for nn-clas
    if nn_metrics['accuracy']:
      dataset_meta = metadata_dict.get(dataset, {})
      avg_train = np.mean(nn_metrics['train_time']).item() if nn_metrics['train_time'] else 0.0
      avg_pred = np.mean(nn_metrics['pred_time']).item() if nn_metrics['pred_time'] else 0.0
      avg_support = np.mean(nn_metrics['support_count']).item() if nn_metrics['support_count'] else 0.0
      results.append({
        'Dataset': dataset,
        'Model': 'nn-clas',
        'k': '',
        'Accuracy': np.mean(nn_metrics['accuracy']),
        'TrainTime': avg_train,
        'PredTime': avg_pred,
        'SupportCount': avg_support
      })

    # Compute averages for knn-clas
    for k in knn_ks:
      if knn_metrics['k_metrics'][k]['accuracy']:
        dataset_meta = metadata_dict.get(dataset, {})
        avg_train = np.mean(knn_metrics['train_time']).item() if knn_metrics['train_time'] else 0.0
        avg_pred = np.mean(knn_metrics['k_metrics'][k]['pred_time']).item() if knn_metrics['k_metrics'][k]['pred_time'] else 0.0
        avg_support = np.mean(knn_metrics['support_count']).item() if knn_metrics['support_count'] else 0.0
        results.append({
          'Dataset': dataset,
          'Model': 'knn-clas',
          'k': k,
          'Accuracy': np.mean(knn_metrics['k_metrics'][k]['accuracy']),
          'TrainTime': avg_train,
          'PredTime': avg_pred,
          'SupportCount': avg_support
        })

  # Print results
  print("\nComparison Results:")
  print("{:<20} {:<10} {:<5} {:<10} {:<10} {:<10} {:<10}".format(
    "Dataset", "Model", "k", "Accuracy", "TrainTime", "PredTime", "SupportCnt"))
  for res in results:
    print("{:<20} {:<10} {:<5} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
      res['Dataset'], res['Model'], res['k'],
      res['Accuracy'], res['TrainTime'], res['PredTime'], int(res['SupportCount'])
    ))

  # Write results to CSV
  output_dir = script_dir / "comparison_results"
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / "setsresults.csv"
  with open(output_file, "w") as f:
    f.write("Dataset,Model,k,Accuracy,TrainTime,PredTime,SupportCount\n")
    for res in results:
      line = (
        f"{res['Dataset']},"
        f"{res['Model']},"
        f"{res['k']},"
        f"{res['Accuracy']:.2f},"
        f"{res['TrainTime']:.2f},"
        f"{res['PredTime']:.2f},"
        f"{int(res['SupportCount'])},\n"
      )
      f.write(line)

if __name__ == "__main__":
  main()
