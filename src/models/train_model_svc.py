import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
import os

def main():
    processed_data_path = "data/processed"
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)

    x_train = pd.read_csv(f"{processed_data_path}/x_train.csv")
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv")
    x_test = pd.read_csv(f"{processed_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv")

    # model = SVC(probability=True)
    # model.fit(x_train, y_train)

    params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    model = SVC(probability=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='recall',
        cv=5,
        verbose=2
    )

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    y_predict = grid_search.predict(x_test)
    print(classification_report(y_test, y_predict))

    model_path = f"{models_path}/svc_model.joblib"
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()

