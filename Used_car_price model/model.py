import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 读取清洗后的数据
df = pd.read_csv("cleaned_car_data.csv")

# 特征选择
features = [
    'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS',
    'model', 'kilometer', 'monthOfRegistration', 'fuelType',
    'brand', 'notRepairedDamage'
]
target = 'price'

X = df[features]
y = df[target]

#  拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  构建模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  模型预测与评估
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f" 模型评估结果：")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 6. 保存模型
joblib.dump(model, "car_price_predictor.pkl")
print(" 模型已保存为 car_price_predictor.pkl")

