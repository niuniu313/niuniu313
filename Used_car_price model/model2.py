import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

#  读取数据
df = pd.read_csv('cleaned_car_data.csv')

#  特征与目标
features = [
    'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS',
    'model', 'kilometer', 'monthOfRegistration',
    'fuelType', 'brand', 'notRepairedDamage'
]
target = 'price'

X = df[features]
y = df[target]

#  拆分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  构建 DMatrix（XGBoost 的输入格式）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#  设置参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.1,
    'max_depth': 6,
    'lambda': 1.0,
    'alpha': 0.0,
    'seed': 42
}

#  模型训练
print("Training XGBoost model...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=50
)

#  模型预测与评估
y_pred = model.predict(dtest)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n 模型评估结果：")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

#  保存模型
joblib.dump(model, 'xgboost_car_price_model.pkl')
print("\n 模型已保存为 xgboost_car_price_model.pkl")
