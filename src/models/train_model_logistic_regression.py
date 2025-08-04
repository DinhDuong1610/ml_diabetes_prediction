import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

def main():
    processed_data_path = "data/processed"
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)

    x_train = pd.read_csv(f"{processed_data_path}/x_train.csv")
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv")
    x_test = pd.read_csv(f"{processed_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv")

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print(classification_report(y_test, y_predict))

    model_path = f"{models_path}/logistic_regression_model.joblib"
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()

