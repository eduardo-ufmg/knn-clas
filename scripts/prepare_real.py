import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

def store_real_dataset(X, y, name):
  """Store dataset in protobuf format with train/test split."""
  # Split into train (80%) and test (20%)
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True
  )
  
  # Create and store training dataset
  train_dataset = create_proto_dataset(X_train, y_train)
  store_dataset(train_dataset, f"data/{name}_train.pb")
  
  # Create and store test samples
  test_samples = create_proto_test_samples(X_test, y_test)
  store_test_samples(test_samples, f"data/{name}_test.pb")

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
    "breast_cancer": load_breast_cancer_dataset(),
    "pima_diabetes": load_pima_diabetes(),
    "haberman": load_haberman_survival(),
    "banknote": load_banknote_authentication(),
    "sonar": load_sonar(),
    # "adult": load_adult_census(),
    "digits_binary": load_digits_binary(),
    "ionosphere": load_ionosphere(),
    "spect_heart": load_spect_heart(),
  }

  for name, (X, y) in datasets.items():
    print(f"Storing {name} dataset...")
    store_real_dataset(X, y, name)

  return datasets

if __name__ == "__main__":
  for name, (X, y) in load_all_datasets().items():
    print(f"{name}: X={X.shape}, y={y.shape}")
