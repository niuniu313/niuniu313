from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# 参数设置
img_size = 64
df = pd.read_csv('dataset.csv')

# 加载图像并缩放
X = []
y = []

for i, row in df.iterrows():
    img_path = os.path.join('data\\part2', row['file'])
    gender = row['gender']
    try:
        img = load_img(img_path, target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        X.append(img)
        y.append(gender)
    except:
        continue

X = np.array(X)
y = np.array(y)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=20,
          batch_size=32,
          callbacks=[early_stop])

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 保存模型
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/gender_classification_model.h5")
