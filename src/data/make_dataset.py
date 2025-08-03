import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/raw/diabetes.csv")

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
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

