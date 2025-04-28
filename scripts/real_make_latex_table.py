import csv
from collections import defaultdict
from pathlib import Path

def csv_to_separate_latex_tables(metadata_csv_path, results_csv_path, output_base_path):
  # Load metadata
  metadata = {}
  with open(metadata_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      metadata[row['name']] = {
        'samples': row['nsamples'],
        'features': row['nfeatures'],
        'class_ratio': row.get('class_ratio', ''),
        'mutual_info': row.get('avg_mutual_info', ''),
        'fisher': row.get('fisher_score', ''),
        'overlap': row.get('overlap_score', ''),
        'imbalance': row.get('imbalance_ratio', '')
      }

  datasets = defaultdict(dict)
  with open(results_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      dataset = row['Dataset'].strip()
      model = row['Model'].strip()
      k = row['k'].strip()
      
      if dataset not in datasets:
        meta = metadata.get(dataset, {})
        datasets[dataset] = {
          'samples': meta.get('samples', ''),
          'features': meta.get('features', ''),
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
        if not datasets[dataset]['knn-clas']['train']:
          datasets[dataset]['knn-clas']['train'] = row['TrainTime']
        datasets[dataset]['knn-clas'][key] = {
          'acc': row['Accuracy'],
          'pred': row['PredTime']
        }

  metadata_table = [
    r"\begin{table}[htbp]",
    r"\caption{Dataset Metadata}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|}",
    r"\hline",
    r"\textbf{Dataset} & \textbf{Samples} & \textbf{Features} \\ \hline"
  ]

  statistics_table = [
    r"\begin{table}[htbp]",
    r"\caption{Dataset Statistics}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|c|c|c|}",
    r"\hline",
    r"\textbf{Dataset} & \textbf{C0/C1} & \textbf{MI} & \textbf{Fisher} & \textbf{Overlap} & \textbf{Imb.Ratio} \\ \hline"
  ]

  accuracy_table = [
    r"\begin{table}[htbp]",
    r"\caption{Model Accuracy Comparison}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c|}{\textbf{Accuracy}} \\ \cline{2-5}",
    r" & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

  timing_table = [
    r"\begin{table}[htbp]",
    r"\caption{Training and Prediction Times}",
    r"\begin{center}",
    r"\begin{tabular}{|c|c|c|c|c|c|c|}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{2}{c|}{\textbf{Training (ms)}} & \multicolumn{4}{c|}{\textbf{Prediction (ms)}} \\ \cline{2-7}",
    r" & \textbf{nn} & \textbf{knn} & \textbf{nn} & \textbf{1nn} & \textbf{3nn} & \textbf{5nn} \\ \hline"
  ]

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
    meta = metadata[dataset]

    metadata_table.append(f"{dataset} & {data['samples']} & {data['features']} \\\\ \\hline")

    statistics_table.append(" & ".join([
      dataset,
      meta.get('class_ratio', ''),
      meta.get('mutual_info', ''),
      meta.get('fisher', ''),
      meta.get('overlap', ''),
      meta.get('imbalance', '')
    ]) + r" \\ \hline")

    accuracy_table.append(" & ".join([
      dataset,
      data['nn-clas']['acc'],
      data['knn-clas']['1nn']['acc'],
      data['knn-clas']['3nn']['acc'],
      data['knn-clas']['5nn']['acc']
    ]) + r" \\ \hline")

    timing_table.append(" & ".join([
      dataset,
      data['nn-clas']['train'],
      data['knn-clas']['train'],
      data['nn-clas']['pred'],
      data['knn-clas']['1nn']['pred'],
      data['knn-clas']['3nn']['pred'],
      data['knn-clas']['5nn']['pred']
    ]) + r" \\ \hline")

    support_table.append(" & ".join([
      dataset,
      data['nn-clas']['support'],
      data['knn-clas']['support']
    ]) + r" \\ \hline")

  for table, name in zip(
    [metadata_table, statistics_table, accuracy_table, timing_table, support_table],
    ['metadata', 'statistics', 'accuracy', 'timing', 'support']
  ):
    table.extend([
      r"\end{tabular}",
      fr"\label{{tab:{name}}}",
      r"\end{center}",
      r"\end{table}"
    ])

  output_path = Path(output_base_path)
  output_files = {
    'metadata': metadata_table,
    'statistics': statistics_table,
    'accuracy': accuracy_table,
    'timing': timing_table,
    'support': support_table
  }

  for key, content in output_files.items():
    file_path = output_path.with_name(f"{key}.tex")
    with open(file_path, 'w') as f:
      f.write("\n".join(content))

if __name__ == "__main__":
  csv_to_separate_latex_tables(
    "scripts/comparison_results/setsmetadata.csv",
    "scripts/comparison_results/setsresults.csv",
    "scripts/comparison_results/setsresults.tex"
  )
