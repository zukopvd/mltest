from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# ----------------------------- #
# 1. Описание архитектуры модели
# ----------------------------- #
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()  # режим инференса

# ----------------------------- #
# 2. FastAPI и входные данные
# ----------------------------- #
app = FastAPI()


class ImageRequest(BaseModel):
    image_base64: str  # поле, в которое будем передавать строку Base64


# ----------------------------- #
# 3. Обрабатываем запрос
# ----------------------------- #
@app.post("/predict")
def predict(data: ImageRequest):
    print(f"Data to predict: {data.image_base64}")
    try:
        # 1. Декодируем Base64 → байты
        image_data = base64.b64decode(data.image_base64)

        # 2. Превращаем байты в изображение (Pillow)
        image = Image.open(BytesIO(image_data)).convert("L")  # "L" - grayscale

        # 3. Изменяем размер на 28x28 (как MNIST), если нужно
        image = image.resize((28, 28))

        # 4. В numpy массив
        img_np = np.array(image, dtype=np.float32)

        # 5. Нормализуем и добавляем размерности для батча и канала: [1, 1, 28, 28]
        img_tensor = torch.tensor(img_np / 255.0).unsqueeze(0).unsqueeze(0)

        # 6. Прогоняем через модель
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            print(f"Predicted: {predicted}")

        return {
            "prediction": int(predicted.item())
        }

    except Exception as e:
        return {"error": str(e)}

