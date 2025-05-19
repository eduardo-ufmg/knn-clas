import numpy as np
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml

def save_dataset(X: np.ndarray, y: np.ndarray, name: str):
    """Helper function to save X and y into the ./sets directory."""
    outdir = './sets'
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f'{name}.npz')
    np.savez_compressed(filepath, X=X, y=y)

def load_breast_cancer_dataset():
    data = load_breast_cancer(return_X_y=False)
    X, y = data.data, data.target
    label_map = {0: -1, 1: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "breast_cancer")
    return X, y

def load_pima_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, header=None, names=cols)
    
    features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for feat in features:
        df[feat] = df[feat].replace(0, np.nan).fillna(df[feat].median())
    
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values
    label_map = {0: -1, 1: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "pima_diabetes")
    return X, y

def load_haberman_survival():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    df = pd.read_csv(url, header=None,
             names=["Age", "YearOfOperation", "PositiveNodes", "SurvivalStatus"])
    y = df["SurvivalStatus"].values
    X = df.iloc[:, :-1].values
    label_map = {1: -1, 2: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "haberman_survival")
    return X, y

def load_banknote_authentication():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, header=None,
             names=["Variance", "Skewness", "Curtosis", "Entropy", "Class"])
    X = df.iloc[:, :-1].values
    y = df["Class"].values
    label_map = {0: -1, 1: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "banknote_authentication")
    return X, y

def load_sonar():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    label_map = {"M": -1, "R": 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "sonar")
    return X, y

def load_adult_census():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    col_names = ["Age","Workclass","fnlwgt","Education","EducationNum","MaritalStatus",
           "Occupation","Relationship","Race","Sex","CapitalGain","CapitalLoss",
           "HoursPerWeek","NativeCountry","Income"]
    df = pd.read_csv(url, header=None, names=col_names, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    X = pd.get_dummies(df.drop("Income", axis=1), drop_first=True).values
    y = df["Income"].values
    label_map = {"<=50K": -1, ">50K": 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "adult_census")
    return X, y

def load_digits_binary():
    digits = load_digits()
    mask = np.logical_or(digits.target == 0, digits.target == 1)
    X = digits.data[mask]
    y = digits.target[mask]
    label_map = {0: -1, 1: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "digits_binary")
    return X, y

def load_ionosphere():
    ionosphere = fetch_openml(data_id=59, parser='auto')
    X = ionosphere.data.to_numpy()
    y = ionosphere.target.to_numpy()
    label_map = {"b": -1, "g": 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "ionosphere")
    return X, y

def load_spect_heart():
    spect = fetch_openml(data_id=337, parser='auto')
    X = spect.data.to_numpy()
    y = spect.target.to_numpy().astype(int)
    label_map = {0: -1, 1: 1}
    y = np.vectorize(label_map.get)(y)
    y = y.reshape(-1, 1)
    save_dataset(X, y, "spect_heart")
    return X, y

def load_all_datasets():
    """Load all datasets and save them to the ./sets directory."""
    datasets = {
        "breast_cancer": load_breast_cancer_dataset,
        "pima_diabetes": load_pima_diabetes,
        "haberman_survival": load_haberman_survival,
        "banknote_authentication": load_banknote_authentication,
        "sonar": load_sonar,
        "adult_census": load_adult_census,
        "digits_binary": load_digits_binary,
        "ionosphere": load_ionosphere,
        "spect_heart": load_spect_heart
    }
    for name, loader in datasets.items():
        print(f"Loading {name} dataset...")
        loader()
        print(f"{name} dataset saved.")

if __name__ == "__main__":
    load_all_datasets()
    print("All datasets loaded and saved.")
    