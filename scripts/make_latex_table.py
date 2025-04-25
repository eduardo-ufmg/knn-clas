import csv
from collections import defaultdict
from pathlib import Path

def csv_to_separate_latex_tables(csv_path, output_base_path):
  datasets = defaultdict(dict)
  with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      dataset = row['Dataset'].strip()
      model = row['Model'].strip()
      k = row['k'].strip()
      
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
      
      if model == 'nn-clas':
        datasets[dataset]['nn-clas'] = {
          'acc': row['Accuracy'],
          'train': row['TrainTime'],
          'pred': row['PredTime'],
          'support': row['SupportCount']
        }
      elif model == 'knn-clas':
        key = f'{k}nn'
        if not datasets[dataset]['knn-clas']['support']:
          datasets[dataset]['knn-clas']['support'] = row['SupportCount']
        datasets[dataset]['knn-clas'][key] = {
          'acc': row['Accuracy'],
          'pred': row['PredTime']
        }
        if not datasets[dataset]['knn-clas']['train']:
          datasets[dataset]['knn-clas']['train'] = row['TrainTime']

  # Generate Dataset Metadata Table
  metadata_table = [
    r"\begin{table}[htbp]",
    r"\caption{Dataset Metadata}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|}",
    r"\hline",
    r"\textbf{Dataset} & \textbf{Samples} & \textbf{Features} \\ \hline"
  ]

  # Generate Accuracy Table
  accuracy_table = [
    r"\begin{table}[htbp]",
    r"\caption{Model Accuracy Comparison}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c|}{\textbf{Accuracy}} \\ \cline{2-5}",
    r" & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

  # Generate Timing Table
  timing_table = [
    r"\begin{table}[htbp]",
    r"\caption{Training and Prediction Times}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{2}{c|}{\textbf{Training (ms)}} & \multicolumn{4}{c|}{\textbf{Prediction (ms)}} \\ \cline{2-7}",
    r" & \textbf{nn} & \textbf{knn} & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

  # Generate Support Count Table
  support_table = [
    r"\begin{table}[htbp]",
    r"\caption{Support Samples Count}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{2}{c|}{\textbf{Support Samples}} \\ \cline{2-3}",
    r" & \textbf{nn} & \textbf{knn} \\ \hline"
  ]

  for dataset, data in datasets.items():

    meta_row = [
      dataset,
      data['samples'],
      data['features']
    ]
    metadata_table.append(" & ".join(meta_row) + r" \\ \hline")

    acc_row = [
      dataset,
      data['nn-clas']['acc'],
      data['knn-clas']['1nn']['acc'],
      data['knn-clas']['3nn']['acc'],
      data['knn-clas']['5nn']['acc']
    ]
    accuracy_table.append(" & ".join(acc_row) + r" \\ \hline")

    time_row = [
      dataset,
      data['nn-clas']['train'],
      data['knn-clas']['train'],
      data['nn-clas']['pred'],
      data['knn-clas']['1nn']['pred'],
      data['knn-clas']['3nn']['pred'],
      data['knn-clas']['5nn']['pred']
    ]
    timing_table.append(" & ".join(time_row) + r" \\ \hline")

    support_row = [
      dataset,
      data['nn-clas']['support'],
      data['knn-clas']['support']
    ]
    support_table.append(" & ".join(support_row) + r" \\ \hline")

  metadata_table.extend([
    r"\end{tabular}",
    r"\label{tab:metadata}",
    r"\end{center}",
    r"\end{table}"
  ])

  accuracy_table.extend([
    r"\end{tabular}",
    r"\label{tab:accuracy}",
    r"\end{center}",
    r"\end{table}"
  ])

  timing_table.extend([
    r"\end{tabular}",
    r"\label{tab:timing}",
    r"\end{center}",
    r"\end{table}"
  ])

  support_table.extend([
    r"\end{tabular}",
    r"\label{tab:support}",
    r"\end{center}",
    r"\end{table}"
  ])

  output_path = Path(output_base_path)
  metadata_path = output_path.with_name(f"{output_path.stem}_metadata.tex")
  accuracy_path = output_path.with_name(f"{output_path.stem}_accuracy.tex")
  timing_path = output_path.with_name(f"{output_path.stem}_timing.tex")
  support_path = output_path.with_name(f"{output_path.stem}_support.tex")

  with open(metadata_path, 'w') as f:
    f.write("\n".join(metadata_table))

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