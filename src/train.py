import os
import yaml
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():
    # 1. Читаем параметры
    params_path = os.path.join(BASE_DIR, "..", "params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)["train"]

    # 2. Читаем подготовленные данные
    data_path = os.path.join(BASE_DIR, "..", "data", "iris.csv")
    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Обучаем модель
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # 4. Сохраняем метрики
    metrics_path = os.path.join(BASE_DIR, "..", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f)

    # 5. Сохраняем модель
    # Убеждаемся, что директория существует, иначе joblib выдаст ошибку
    models_dir = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(model, model_path)

    print(f"Модель обучена. Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
