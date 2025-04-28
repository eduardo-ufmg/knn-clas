import os
import csv
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import mutual_info_score
import classifier_pb2 as pb
from store_proto import store_dataset, store_test_samples

def create_proto_dataset(X, y):
  """Convert numpy arrays to protobuf Dataset message."""
  dataset = pb.Dataset()
  for features, target in zip(X, y):
    entry = dataset.entries.add()
    entry.features.extend(features.tolist())
    if isinstance(target[0], (np.int32, np.int64)):
      entry.target.target_int = int(target[0])
    else:
      entry.target.target_str = str(target[0])
  return dataset

def create_proto_test_samples(X, y):
  """Convert numpy arrays to protobuf TestSamples message."""
  samples = pb.TestSamples()
  for i, (features, target) in enumerate(zip(X, y)):
    entry = samples.entries.add()
    entry.sample_id = i
    entry.features.extend(features.tolist())
    if isinstance(target[0], (np.int32, np.int64)):
      entry.ground_truth.target_int = int(target[0])
    else:
      entry.ground_truth.target_str = str(target[0])
  return samples

def remove_correlated(X_train, X_test, threshold=0.95):
  """Remove highly correlated features based on training data."""
  corr_matrix = np.corrcoef(X_train, rowvar=False)
  upper = np.triu(corr_matrix, k=1)
  to_drop = set()
  for i in range(upper.shape[0]):
    for j in range(i + 1, upper.shape[1]):
      if abs(upper[i, j]) > threshold:
        to_drop.add(j)
  mask = [i for i in range(X_train.shape[1]) if i not in to_drop]
  return X_train[:, mask], X_test[:, mask]

def preprocess_dataset(X_train_raw, X_test_raw, y_train, y_test):
  # 1. Normalize to [-1, 1]
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X_train_scaled = scaler.fit_transform(X_train_raw)
  X_test_scaled = scaler.transform(X_test_raw)

  # 2. Remove low-variance features
  var_selector = VarianceThreshold(threshold=0.01)
  X_train_var = var_selector.fit_transform(X_train_scaled)
  X_test_var = var_selector.transform(X_test_scaled)

  # 3. Remove highly correlated features
  X_train_corr, X_test_corr = remove_correlated(X_train_var, X_test_var, 0.95)

  # 4. Select informative features based on mutual information
  mi = mutual_info_classif(X_train_corr, y_train.ravel(), discrete_features='auto')
  informative_idx = np.where(mi > 0.01)[0]
  X_train_final = X_train_corr[:, informative_idx]
  X_test_final = X_test_corr[:, informative_idx]

  return X_train_final, X_test_final

def store_real_dataset(X, y, name, n_splits=10):

  complete_dataset_to_fit_name = f"data/{name}_complete_fit.pb"
  complete_dataset_to_pred_name = f"data/{name}_complete_pred.pb"

  # Preprocess the complete dataset
  X_preprocessed, X_preprocessed = preprocess_dataset(X, X, y, y)

  store_dataset(create_proto_dataset(X_preprocessed, y), complete_dataset_to_fit_name)
  store_test_samples(create_proto_test_samples(X_preprocessed, y), complete_dataset_to_pred_name)

  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
  for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Preprocess the dataset
    X_train_final, X_test_final = preprocess_dataset(X_train_raw, X_test_raw, y_train, y_test)

    # Store processed datasets
    train_dataset = create_proto_dataset(X_train_final, y_train)
    train_filename = f"data/{name}_fold{fold}_train.pb"
    store_dataset(train_dataset, train_filename)

    test_samples = create_proto_test_samples(X_test_final, y_test)
    test_filename = f"data/{name}_fold{fold}_test.pb"
    store_test_samples(test_samples, test_filename)

def load_breast_cancer_dataset():
  data = load_breast_cancer(return_X_y=False)
  X, y = data.data, data.target
  label_map = {0: "Malignant", 1: "Benign"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_pima_diabetes():
  url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
  )
  cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
  ]
  df = pd.read_csv(url, header=None, names=cols)
  
  features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
  
  for feat in features:
    df[feat] = df[feat].replace(0, np.nan)
    df[feat] = df[feat].fillna(df[feat].median())
  
  X = df.drop("Outcome", axis=1).values
  y = df["Outcome"].values
  label_map = {0: "Non-Diabetic", 1: "Diabetic"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_haberman_survival():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"

  df = pd.read_csv(url, header=None,
                    names=["Age", "YearOfOperation", "PositiveNodes", "SurvivalStatus"])
  
  y = df["SurvivalStatus"].values
  X = df.iloc[:, :-1].values
  label_map = {1: "Survived", 2: "Not Survived"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_banknote_authentication():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

  df = pd.read_csv(url, header=None,
                    names=["Variance", "Skewness", "Curtosis", "Entropy", "Class"])
  
  X = df.iloc[:, :-1].values
  y = df["Class"].values
  label_map = {0: "Genuine", 1: "Forged"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_sonar():
  url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
          "undocumented/connectionist-bench/sonar/sonar.all-data")
  
  df = pd.read_csv(url, header=None)

  # Last column is 'M' or 'R'
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  label_map = {"M": "Mine", "R": "Rock"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_adult_census():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

  col_names = ["Age","Workclass","fnlwgt","Education","EducationNum","MaritalStatus",
                "Occupation","Relationship","Race","Sex","CapitalGain","CapitalLoss",
                "HoursPerWeek","NativeCountry","Income"]
  
  df = pd.read_csv(url, header=None, names=col_names,
                    na_values=" ?", skipinitialspace=True)
  
  df.dropna(inplace=True)

  # Keep Income as is and encode categoricals
  X = pd.get_dummies(df.drop("Income", axis=1), drop_first=True).values
  y = df["Income"].values

  return X, y.reshape(-1, 1)

def load_digits_binary():
  digits = load_digits()

  mask = np.logical_or(digits.target == 0, digits.target == 1)
  X = digits.data[mask]
  y = digits.target[mask]

  return X, y.reshape(-1, 1)

def load_ionosphere():
  ionosphere = fetch_openml(data_id=59, parser='auto')

  X = ionosphere.data.to_numpy()
  y = ionosphere.target.to_numpy()
  label_map = {"b": "Bad", "g": "Good"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_spect_heart():
  spect = fetch_openml(data_id=337, parser='auto')

  X = spect.data.to_numpy()
  y = spect.target.to_numpy().astype(int)
  label_map = {0: "Normal", 1: "Abnormal"}
  y = np.vectorize(label_map.get)(y)
  return X, y.reshape(-1, 1)

def load_all_datasets():
  datasets = {
    "Breast Cancer": load_breast_cancer_dataset(),
    "Pima Diabetes": load_pima_diabetes(),
    "Haberman": load_haberman_survival(),
    "Banknote": load_banknote_authentication(),
    "Sonar": load_sonar(),
    "Binary Digits": load_digits_binary(),
    "Ionosphere": load_ionosphere(),
    "SPECT Heart": load_spect_heart(),
  }

  for name, (X, y) in datasets.items():
    print(f"Storing {name} dataset with 10 folds...")
    store_real_dataset(X, y, name)

  return datasets

def compute_statistics(X, y):
  y_flat = y.ravel()
  unique_classes = np.unique(y_flat)
  if len(unique_classes) != 2:
    raise ValueError("The dataset must have exactly two unique classes.")
  
  class0, class1 = unique_classes
  class0_mask = y_flat == class0
  class1_mask = y_flat == class1
  
  class_counts = [np.sum(class0_mask), np.sum(class1_mask)]
  class_ratio = class_counts[0] / class_counts[1] if class_counts[1] != 0 else np.inf

  # Average mutual information
  avg_mi = mutual_info_classif(X, y_flat, discrete_features='auto').mean()

  # Fisher score: mean diff squared over var sum
  mean0 = X[class0_mask].mean(axis=0)
  mean1 = X[class1_mask].mean(axis=0)
  var0 = X[class0_mask].var(axis=0)
  var1 = X[class1_mask].var(axis=0)
  fisher_scores = (mean0 - mean1)**2 / (var0 + var1 + 1e-6)
  avg_fisher = np.mean(fisher_scores)

  # Overlap score: average number of features where class means are within 1 std
  std0 = X[class0_mask].std(axis=0)
  std1 = X[class1_mask].std(axis=0)
  overlap = np.mean(np.abs(mean0 - mean1) < (std0 + std1) / 2)

  # Imbalance ratio: max(counts) / min(counts)
  imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) != 0 else np.inf

  return class_ratio, avg_mi, avg_fisher, overlap, imbalance_ratio

if __name__ == "__main__":
  os.makedirs("scripts/comparison_results", exist_ok=True)

  metadata = []
  for name, (X, y) in load_all_datasets().items():
    nsamples, nfeatures = X.shape
    class_ratio, avg_mi, avg_fisher, overlap, imbalance_ratio = compute_statistics(X, y)
    metadata.append({
      "name": name,
      "nsamples": nsamples,
      "nfeatures": nfeatures,
      "class_ratio": round(class_ratio, 2),
      "avg_mutual_info": round(avg_mi, 2),
      "fisher_score": round(avg_fisher, 2),
      "overlap_score": round(overlap, 2),
      "imbalance_ratio": round(imbalance_ratio, 2),
    })

  csv_path = "scripts/comparison_results/setsmetadata.csv"
  with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
      "name", "nsamples", "nfeatures",
      "class_ratio", "avg_mutual_info", "fisher_score",
      "overlap_score", "imbalance_ratio"
    ])
    writer.writeheader()
    writer.writerows(metadata)

  print(f"Extended metadata saved to {csv_path}")

