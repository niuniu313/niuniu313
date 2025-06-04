from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# 加载模型
model = load_model('model/gender_classification_model.h5')

# 设置图像大小
img_size = 64

def predict_gender(img_path):
    try:
        img = load_img(img_path, target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # 扩展成 batch 维度
        pred = model.predict(img)[0][0]
        gender = 'Female' if pred >= 0.5 else 'Male'
        print(f"Predicted Gender: {gender} (score: {pred:.4f})")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 假设你有一张图片放在 test.jpg
predict_gender('jpg_path')
