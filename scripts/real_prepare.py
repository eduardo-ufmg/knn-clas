import os
import csv
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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

def store_real_dataset(X, y, name, n_splits=10):
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
  for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Save training dataset for this fold
    train_dataset = create_proto_dataset(X_train, y_train)
    train_filename = f"data/{name}_fold{fold}_train.pb"
    store_dataset(train_dataset, train_filename)
    
    # Save test samples for this fold
    test_samples = create_proto_test_samples(X_test, y_test)
    test_filename = f"data/{name}_fold{fold}_test.pb"
    store_test_samples(test_samples, test_filename)

def load_breast_cancer_dataset():
  data = load_breast_cancer(return_X_y=False)
  X, y = data.data, data.target
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
  
  # Pattern 1: direct assignment instead of inplace on a slice
  for feat in features:
    df[feat] = df[feat].replace(0, np.nan)
    df[feat] = df[feat].fillna(df[feat].median())
  
  X = df.drop("Outcome", axis=1).values
  y = df["Outcome"].values.reshape(-1, 1)
  return X, y

def load_haberman_survival():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"

  df = pd.read_csv(url, header=None,
                    names=["Age","YearOfOperation","PositiveNodes","SurvivalStatus"])
  
  # Map SurvivalStatus: 1 → survived ≥5 yrs (1), 2 → died within 5 yrs (0)
  y = df["SurvivalStatus"].map({1: 1, 2: 0}).values
  X = df.iloc[:, :-1].values

  return X, y.reshape(-1, 1)

def load_banknote_authentication():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

  df = pd.read_csv(url, header=None,
                    names=["Variance","Skewness","Curtosis","Entropy","Class"])
  
  X = df.iloc[:, :-1].values
  y = df["Class"].values
  
  return X, y.reshape(-1, 1)

def load_sonar():
  url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
          "undocumented/connectionist-bench/sonar/sonar.all-data")
  
  df = pd.read_csv(url, header=None)

  # Last column is 'M' or 'R'
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].map({'M': 1, 'R': 0}).values

  return X, y.reshape(-1, 1)

def load_adult_census():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

  col_names = ["Age","Workclass","fnlwgt","Education","EducationNum","MaritalStatus",
                "Occupation","Relationship","Race","Sex","CapitalGain","CapitalLoss",
                "HoursPerWeek","NativeCountry","Income"]
  
  df = pd.read_csv(url, header=None, names=col_names,
                    na_values=" ?", skipinitialspace=True)
  
  df.dropna(inplace=True)
  # Encode Income and categoricals
  df["Income"] = df["Income"].map({"<=50K": 0, ">50K": 1})
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
  y = LabelEncoder().fit_transform(ionosphere.target)

  return X, y.reshape(-1, 1)

def load_spect_heart():
  spect = fetch_openml(data_id=337, parser='auto')

  X = spect.data.to_numpy()
  y = spect.target.to_numpy().astype(int)

  return X, y.reshape(-1, 1)

def load_all_datasets():
  datasets = {
    "Breast Cancer": load_breast_cancer_dataset(),
    "Pima Diabetes": load_pima_diabetes(),
    "Haberman": load_haberman_survival(),
    "Banknote": load_banknote_authentication(),
    "Sonar": load_sonar(),
    # "Adult": load_adult_census(),
    "Binary Digits": load_digits_binary(),
    "Ionosphere": load_ionosphere(),
    "SPECT Heart": load_spect_heart(),
  }

  for name, (X, y) in datasets.items():
    print(f"Storing {name} dataset with 10 folds...")
    store_real_dataset(X, y, name)

  return datasets

if __name__ == "__main__":
  # Ensure the directory exists
  os.makedirs("scripts/comparison_results", exist_ok=True)

  # Prepare metadata
  metadata = []
  for name, (X, y) in load_all_datasets().items():
    nsamples, nfeatures = X.shape
    metadata.append({"name": name, "nsamples": nsamples, "nfeatures": nfeatures})

  # Write metadata to CSV
  csv_path = "scripts/comparison_results/setsmetadata.csv"
  with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["name", "nsamples", "nfeatures"])
    writer.writeheader()
    writer.writerows(metadata)

  print(f"Metadata saved to {csv_path}")
