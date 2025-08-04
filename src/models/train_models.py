import pandas as pd
from lazypredict.Supervised import LazyClassifier
import os

processed_data_path = "data/processed"
models_path = "models"
os.makedirs(models_path, exist_ok=True)

x_train = pd.read_csv(f"{processed_data_path}/x_train.csv")
y_train = pd.read_csv(f"{processed_data_path}/y_train.csv")
x_test = pd.read_csv(f"{processed_data_path}/x_test.csv")
y_test = pd.read_csv(f"{processed_data_path}/y_test.csv")

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)

print(models)

