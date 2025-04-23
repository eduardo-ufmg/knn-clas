import csv
from collections import defaultdict

def csv_to_latex_table(csv_path, output_path):
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
          'samples': row['nSamples'].strip(),
          'features': row['nFeatures'].strip(),
          'models': {
            'nn-clas': {'acc': None, 'train_time': None, 'pred_time': None},
            'knn-clas': {
              'train_time': None,
              'k_metrics': {
                1: {'acc': None, 'pred_time': None},
                3: {'acc': None, 'pred_time': None},
                5: {'acc': None, 'pred_time': None}
              }
            }
          }
        }
      
      # Populate data
      if model == 'nn-clas':
        datasets[dataset]['models']['nn-clas']['acc'] = row['Accuracy'].strip()
        datasets[dataset]['models']['nn-clas']['train_time'] = row['TrainTime'].strip()
        datasets[dataset]['models']['nn-clas']['pred_time'] = row['PredTime'].strip()
      elif model == 'knn-clas':
        k_int = int(k)
        datasets[dataset]['models']['knn-clas']['k_metrics'][k_int]['acc'] = row['Accuracy'].strip()
        datasets[dataset]['models']['knn-clas']['k_metrics'][k_int]['pred_time'] = row['PredTime'].strip()
        # Set training time once (same for all k)
        if datasets[dataset]['models']['knn-clas']['train_time'] is None:
          datasets[dataset]['models']['knn-clas']['train_time'] = row['TrainTime'].strip()

  # Generate LaTeX table
  latex = [
    r"\begin{table}[H]",
    r"\centering",
    r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Samples}} & \multirow{2}{*}{\textbf{Features}} & \multicolumn{4}{c|}{\textbf{Accuracy}} & \multicolumn{2}{c|}{\textbf{Training Time (ms)}} & \multicolumn{4}{c|}{\textbf{Prediction Time (ms)}} \\ \cline{4-7} \cline{8-9} \cline{10-13}",
    r" & & & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} & \textbf{nn} & \textbf{knn} & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

  # Add data rows
  for dataset, data in datasets.items():
    row = [
      dataset,
      data['samples'],
      data['features'],
      # Accuracy
      data['models']['nn-clas']['acc'],
      data['models']['knn-clas']['k_metrics'][1]['acc'],
      data['models']['knn-clas']['k_metrics'][3]['acc'],
      data['models']['knn-clas']['k_metrics'][5]['acc'],
      # Training Time
      data['models']['nn-clas']['train_time'],
      data['models']['knn-clas']['train_time'],
      # Prediction Time
      data['models']['nn-clas']['pred_time'],
      data['models']['knn-clas']['k_metrics'][1]['pred_time'],
      data['models']['knn-clas']['k_metrics'][3]['pred_time'],
      data['models']['knn-clas']['k_metrics'][5]['pred_time'],
    ]
    latex.append(" & ".join(row) + r" \\ \hline")

  # Close table
  latex.extend([
    r"\end{tabular}",
    r"\caption{Comparison of Models with Training and Prediction Times}",
    r"\label{tab:comparison_results}",
    r"\end{table}"
  ])

  # Write output
  with open(output_path, 'w') as f:
    f.write("\n".join(latex))

if __name__ == "__main__":
  csv_to_latex_table("scripts/comparison_results/real_sets.csv", "scripts/comparison_results/real_sets_results.tex")