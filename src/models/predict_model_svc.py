import joblib
import pandas as pd

def predict_new_data(data):
    models_path = "models"
    processed_data_path = "data/processed"

    try:
        model = joblib.load(f"{models_path}/svc_model.joblib")
        scaler = joblib.load(f"{processed_data_path}/scaler.joblib")

        train_cols = pd.read_csv(f"{processed_data_path}/x_train.csv").columns
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp model hoặc scaler. Vui lòng chạy script training trước.")
        return None, None

    df = pd.DataFrame([data])
    df = df[train_cols]

    scaled_data_array = scaler.transform(df)

    scaled_data = pd.DataFrame(scaled_data_array, columns=df.columns)

    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)

    return prediction[0], probabilities[0]

if __name__ == '__main__':
    new_patient_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }

    prediction, probabilities = predict_new_data(new_patient_data)

    if prediction is not None:
        print("--- KẾT QUẢ DỰ ĐOÁN ---")
        if prediction == 1:
            print("Dự đoán: Bệnh nhân CÓ nguy cơ bị tiểu đường.")
        else:
            print("Dự đoán: Bệnh nhân KHÔNG có nguy cơ bị tiểu đường.")

        print(f"Tỷ lệ xác suất (Không bị / Bị): {probabilities[0]:.2f} / {probabilities[1]:.2f}")
