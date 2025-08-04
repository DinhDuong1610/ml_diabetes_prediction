import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import os

def main():
    processed_data_path = "data/processed"
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)

    x_train = pd.read_csv(f"{processed_data_path}/x_train.csv")
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv").values.ravel()
    x_test = pd.read_csv(f"{processed_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv").values.ravel()

    # model = RandomForestClassifier(n_estimators=200, criterion="gini", random_state=42)
    # model.fit(x_train, y_train)

    params = {
        'n_estimators': [100, 200, 300, 400],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=params, scoring="recall", cv=5, verbose=2)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    y_predict = grid_search.predict(x_test)
    print(classification_report(y_test, y_predict))

    model_path = f"{models_path}/random_forest_model.joblib"
    joblib.dump(grid_search, model_path)

if __name__ == '__main__':
    main()

