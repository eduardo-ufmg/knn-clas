import csv
from collections import defaultdict

def csv_to_latex_tables(csv_path, output_path):
  # Read and organize data
  datasets = defaultdict(dict)
  with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      dataset = row['Dataset'].strip()
      model = row['Model'].strip()
      k = row['k'].strip()
      
      # Initialize dataset entry
      if dataset not in datasets:
        datasets[dataset] = {
          'samples': row['nSamples'],
          'features': row['nFeatures'],
          'nn-clas': {'acc': None, 'train': None, 'pred': None},
          'knn-clas': {
            'train': None,
            '1nn': {'acc': None, 'pred': None},
            '3nn': {'acc': None, 'pred': None},
            '5nn': {'acc': None, 'pred': None}
          }
        }
      
      # Populate data
      if model == 'nn-clas':
        datasets[dataset]['nn-clas'] = {
          'acc': row['Accuracy'],
          'train': row['TrainTime'],
          'pred': row['PredTime']
        }
      elif model == 'knn-clas':
        key = f'{k}nn'
        datasets[dataset]['knn-clas'][key] = {
          'acc': row['Accuracy'],
          'pred': row['PredTime']
        }
        # Store training time once
        if not datasets[dataset]['knn-clas']['train']:
          datasets[dataset]['knn-clas']['train'] = row['TrainTime']

  # Generate Accuracy Table
  accuracy_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\begin{tabular}{|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Samples}} & \multirow{2}{*}{\textbf{Features}} & \multicolumn{4}{c|}{\textbf{Accuracy}} \\ \cline{4-7}",
    r" & & & \textbf{nn-clas} & \textbf{1nn-clas} & \textbf{3nn-clas} & \textbf{5nn-clas} \\ \hline"
  ]

  # Generate Timing Table
  timing_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Samples}} & \multirow{2}{*}{\textbf{Features}} & \multicolumn{2}{c|}{\textbf{Training (ms)}} & \multicolumn{4}{c|}{\textbf{Prediction (ms)}} \\ \cline{4-9}",
    r" & & & \textbf{nn-clas} & \textbf{knn-clas} & \textbf{nn-clas} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

  # Add data rows to both tables
  for dataset, data in datasets.items():
    # Accuracy table row
    acc_row = [
      dataset,
      data['samples'],
      data['features'],
      data['nn-clas']['acc'],
      data['knn-clas']['1nn']['acc'],
      data['knn-clas']['3nn']['acc'],
      data['knn-clas']['5nn']['acc']
    ]
    accuracy_table.append(" & ".join(acc_row) + r" \\ \hline")

    # Timing table row
    time_row = [
      dataset,
      data['samples'],
      data['features'],
      data['nn-clas']['train'],
      data['knn-clas']['train'],
      data['nn-clas']['pred'],
      data['knn-clas']['1nn']['pred'],
      data['knn-clas']['3nn']['pred'],
      data['knn-clas']['5nn']['pred']
    ]
    timing_table.append(" & ".join(time_row) + r" \\ \hline")

  # Close tables
  for table in [accuracy_table, timing_table]:
    table.extend([
      r"\end{tabular}",
      r"\caption{" + ("Model Accuracy Comparison" if table == accuracy_table else "Training and Prediction Times") + "}",
      r"\label{tab:" + ("accuracy" if table == accuracy_table else "timing") + "}",
      r"\end{table}"
    ])

  # Combine both tables
  full_output = "\n\n".join(["\n".join(accuracy_table), "\n".join(timing_table)])

  # Write output
  with open(output_path, 'w') as f:
    f.write(full_output)

if __name__ == "__main__":
  csv_to_latex_tables("scripts/comparison_results/real_sets.csv", 
             "scripts/comparison_results/real_sets_results.tex")