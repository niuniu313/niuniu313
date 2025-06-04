# 🚗 二手车价格预测模型

本项目基于清洗后的二手车数据，使用 **随机森林回归模型（Random Forest Regressor）** 来预测汽车价格。该模型能够有效处理非线性关系，并适应不同类型的变量。

---

## 📁 项目结构

```bash
.
├── data/
│   └── cleaned_car_data.csv      # 清洗后的训练数据
├── model/
│   └── random_forest_model.pkl   # 训练好的模型（可选）
├── notebook/
│   └── car_price_prediction.ipynb # Jupyter Notebook 代码
├── README.md                     # 项目说明文档

