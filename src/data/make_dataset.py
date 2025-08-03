import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    raw_data_path = "data/raw/diabetes.csv"
    processed_data_path = "data/processed"

    os.makedirs(processed_data_path, exist_ok=True)

    data = pd.read_csv(raw_data_path)

    result = data.describe()
    print(result)

    info = data.info()
    print(info)

    # profile = ProfileReport(data, title="Diabets report", explorative=True)
    # profile.to_file("reports/report.html")

    target = "Outcome"
    x = data.drop(target, axis=1)
    y = data[target]

    print(y.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    print(y_train.value_counts())
    print(y_test.value_counts())

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train_scaled, columns=x.columns)
    x_test = pd.DataFrame(x_test_scaled, columns=x.columns)

    joblib.dump(scaler, f"{processed_data_path}/scaler.joblib")
    x_train.to_csv(f"{processed_data_path}/x_train.csv", index=False)
    x_test.to_csv(f"{processed_data_path}/x_test.csv", index=False)
    y_train.to_csv(f"{processed_data_path}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_data_path}/y_test.csv", index=False)

if __name__ == '__main__':
    main()

