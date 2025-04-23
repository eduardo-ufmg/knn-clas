import csv
from collections import defaultdict
from pathlib import Path

def csv_to_separate_latex_tables(csv_path, output_base_path):
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
          'nn-clas': {'acc': None, 'train': None, 'pred': None, 'support': None},
          'knn-clas': {
            'train': None,
            'support': None,
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
          'pred': row['PredTime'],
          'support': row['SupportCount']
        }
      elif model == 'knn-clas':
        key = f'{k}nn'
        # Only store support count once
        if not datasets[dataset]['knn-clas']['support']:
          datasets[dataset]['knn-clas']['support'] = row['SupportCount']
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

  # Generate Support Count Table
  support_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\begin{tabular}{|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Samples}} & \multirow{2}{*}{\textbf{Features}} & \multicolumn{2}{c|}{\textbf{Support Samples}} \\ \cline{4-5}",
    r" & & & \textbf{nn-clas} & \textbf{knn-clas} \\ \hline"
  ]

  # Add data rows to all tables
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

    # Support count table row
    support_row = [
      dataset,
      data['samples'],
      data['features'],
      data['nn-clas']['support'],
      data['knn-clas']['support']
    ]
    support_table.append(" & ".join(support_row) + r" \\ \hline")

  # Close tables
  accuracy_table.extend([
    r"\end{tabular}",
    r"\caption{Model Accuracy Comparison}",
    r"\label{tab:accuracy}",
    r"\end{table}"
  ])

  timing_table.extend([
    r"\end{tabular}",
    r"\caption{Training and Prediction Times}",
    r"\label{tab:timing}",
    r"\end{table}"
  ])

  support_table.extend([
    r"\end{tabular}",
    r"\caption{Support Samples Count}",
    r"\label{tab:support}",
    r"\end{table}"
  ])

  # Write to separate files
  output_path = Path(output_base_path)
  accuracy_path = output_path.with_name(f"{output_path.stem}_accuracy.tex")
  timing_path = output_path.with_name(f"{output_path.stem}_timing.tex")
  support_path = output_path.with_name(f"{output_path.stem}_support.tex")

  with open(accuracy_path, 'w') as f:
    f.write("\n".join(accuracy_table))
  
  with open(timing_path, 'w') as f:
    f.write("\n".join(timing_table))

  with open(support_path, 'w') as f:
    f.write("\n".join(support_table))

if __name__ == "__main__":
  csv_to_separate_latex_tables(
    "scripts/comparison_results/real_sets.csv",
    "scripts/comparison_results/real_sets_results.tex"
  )
  