import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# 1. Определяем абсолютные пути (чтобы избежать ошибок при деплое)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

# 2. Безопасная загрузка модели ОДИН раз при старте сервера
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Модель не найдена по пути: {MODEL_PATH}. Убедитесь, что этап обучения пройден."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris ML API", description="API для предсказания сорта Ириса")


# 3. Редиректим нас сразу на главную страницу (Swagger UI)
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


# 4. Контракт данных (Pydantic).
# Добавлена строгая валидация (ge=0), аналогично вашим проверкам в схеме данных!
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, description="Длина чашелистика (см)")
    sepal_width: float = Field(..., ge=0.0, description="Ширина чашелистика (см)")
    petal_length: float = Field(..., ge=0.0, description="Длина лепестка (см)")
    petal_width: float = Field(..., ge=0.0, description="Ширина лепестка (см)")


# 5. Эндпоинт для проверки здоровья сервера (Uptime Monitor)
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}


# 6. Эндпоинт для предсказания (принимает JSON, возвращает JSON)
@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # Преобразуем входящие данные в массив для scikit-learn
        data = [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]

        prediction = model.predict(data)

        # Возвращаем результат
        return {"predicted_class": int(prediction[0])}

    except Exception as e:
        # Отлавливаем непредвиденные ошибки модели (например, несовпадение версий scikit-learn)
        raise HTTPException(status_code=500, detail=str(e))
