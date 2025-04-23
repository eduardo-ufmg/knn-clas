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
      
      # Store common dataset info
      if dataset not in datasets:
        datasets[dataset] = {
          'samples': row['nSamples'].strip(),
          'features': row['nFeatures'].strip(),
          'models': {}
        }
      
      # Determine model identifier
      if model == 'nn-clas':
        model_key = 'nn-clas'
      else:
        model_key = f'{k}nn-clas'
      
      # Store metrics for this model
      datasets[dataset]['models'][model_key] = {
        'acc': row['Accuracy'].strip()
      }

  # Generate LaTeX table
  latex = [
    r"\begin{table}[H]",
    r"\centering",
    r"\begin{tabular}{|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Samples}} & \multirow{2}{*}{\textbf{Features}} & \multicolumn{4}{c|}{\textbf{Accuracy}} \\ \cline{4-7}",
    r" & & & \textbf{nn-clas} & \textbf{1nn-clas} & \textbf{3nn-clas} & \textbf{5nn-clas} \\ \hline"
  ]

  # Add data rows
  model_order = ['nn-clas', '1nn-clas', '3nn-clas', '5nn-clas']
  for dataset, data in datasets.items():
    row = [
      f"{dataset}",
      f"{data['samples']}",
      f"{data['features']}"
    ]
    
    # Add metrics in order: acc
    for metric in ['acc']:
      for model in model_order:
        row.append(data['models'].get(model, {}).get(metric, '...'))
    
    latex.append(" & ".join(row) + r" \\ \hline")

  # Add final rows
  latex.extend([
    r"\end{tabular}",
    r"\caption{Comparison of Models on Different Datasets}",
    r"\label{tab:comparison_results}",
    r"\end{table}"
  ])

  # Write output
  with open(output_path, 'w') as f:
    f.write("\n".join(latex))

if __name__ == "__main__":
  csv_to_latex_table("scripts/comparison_results/real_sets.csv", "scripts/comparison_results/real_sets_results.tex")
