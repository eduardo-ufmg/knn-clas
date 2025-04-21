import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

# Scikit-learn datasets (already preprocessed)
def load_sklearn_iris():
  data = load_iris()
  return data.data, data.target.reshape(-1, 1)

def load_sklearn_digits():
  data = load_digits()
  return data.data, data.target.reshape(-1, 1)

def load_sklearn_wine():
  data = load_wine()
  return data.data, data.target.reshape(-1, 1)

def load_sklearn_breast_cancer():
  data = load_breast_cancer()
  return data.data, data.target.reshape(-1, 1)

# UCI Repository datasets (require preprocessing)
def load_uci_spambase():
  spambase = fetch_ucirepo(id=94)
  X = spambase.data.features.to_numpy()
  y = spambase.data.targets.to_numpy().reshape(-1, 1)
  return X, y

def load_uci_adult():
  # Fetch dataset
  adult = fetch_ucirepo(id=2)
  X_df = adult.data.features.copy()
  y_df = adult.data.targets.copy()
  
  # Explicitly define numerical/categorical columns (metadata might be unreliable)
  numerical_vars = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
  categorical_vars = ['workclass', 'education', 'marital-status', 'occupation', 
                      'relationship', 'race', 'sex', 'native-country']
  
  # Handle missing values
  X_df.replace('?', np.nan, inplace=True)
  
  # Process numerical features
  X_num = X_df[numerical_vars].astype(float)
  num_imputer = SimpleImputer(strategy='mean')
  X_num = num_imputer.fit_transform(X_num)
  
  # Process categorical features
  cat_imputer = SimpleImputer(strategy='most_frequent')
  X_cat = cat_imputer.fit_transform(X_df[categorical_vars])
  
  encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_cat_encoded = encoder.fit_transform(X_cat)
  
  # Combine features
  X = np.hstack([X_num, X_cat_encoded])
  
  # Process target variable
  y = y_df.iloc[:, 0].str.replace('.', '', regex=False)
  y = y.str.strip().map({'<=50K': 0, '>50K': 1}).astype(int)
  y = y.to_numpy().reshape(-1, 1)
  
  return X.astype(float), y

# Dictionary of available datasets
DATASET_LOADERS = {
  'iris': load_sklearn_iris,
  'digits': load_sklearn_digits,
  'wine': load_sklearn_wine,
  'breast_cancer': load_sklearn_breast_cancer,
  'spambase': load_uci_spambase,
  'adult': load_uci_adult,
}

# Example usage:
if __name__ == "__main__":
  for dataset_name in DATASET_LOADERS.keys():
    X, y = DATASET_LOADERS[dataset_name]()
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Feature type: {type(X)}, Target type: {type(y)}")
    print(f"First 3 targets: {y[:3].flatten()}")