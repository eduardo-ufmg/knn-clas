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
                'acc': row['Accuracy'].strip(),
                'pre': row['Precision'].strip(),
                'rec': row['Recall'].strip(),
                'f1': row['F1'].strip()
            }

    # Generate LaTeX table
    latex = [
        r"\begin{table}[H]",
        r"\begin{tabular}{cccllllllllllllllll}",
        r"\multirow{2}{*}{dataset} & \multirow{2}{*}{samples} & \multirow{2}{*}{features} & ",
        r"\multicolumn{4}{c}{accuracy} & \multicolumn{4}{c}{precision} & ",
        r"\multicolumn{4}{c}{recall} & \multicolumn{4}{c}{f1} \\",
        r" & & & ",
        r"\multicolumn{1}{c}{nn-clas} & \multicolumn{1}{c}{1nn-clas} & ",
        r"\multicolumn{1}{c}{3nn-clas} & \multicolumn{1}{c}{5nn-clas} & ",
        r"\multicolumn{1}{c}{nn-clas} & \multicolumn{1}{c}{1nn-clas} & ",
        r"\multicolumn{1}{c}{3nn-clas} & \multicolumn{1}{c}{5nn-clas} & ",
        r"\multicolumn{1}{c}{nn-clas} & \multicolumn{1}{c}{1nn-clas} & ",
        r"\multicolumn{1}{c}{3nn-clas} & \multicolumn{1}{c}{5nn-clas} & ",
        r"\multicolumn{1}{c}{nn-clas} & \multicolumn{1}{c}{1nn-clas} & ",
        r"\multicolumn{1}{c}{3nn-clas} & \multicolumn{1}{c}{5nn-clas} \\"
    ]

    # Add data rows
    model_order = ['nn-clas', '1nn-clas', '3nn-clas', '5nn-clas']
    for dataset, data in datasets.items():
        row = [
            f"{dataset}",
            f"{data['samples']}",
            f"{data['features']}"
        ]
        
        # Add metrics in order: acc, pre, rec, f1
        for metric in ['acc', 'pre', 'rec', 'f1']:
            for model in model_order:
                row.append(data['models'].get(model, {}).get(metric, '...'))
        
        latex.append(" & ".join(row) + r" \\")

    # Add final rows
    latex.extend([
        r"\end{tabular}",
        r"\end{table}"
    ])

    # Write output
    with open(output_path, 'w') as f:
        f.write("\n".join(latex))

if __name__ == "__main__":
    csv_to_latex_table("scripts/comparison_results/real_sets.csv", "scripts/comparison_results/real_sets_results.tex")