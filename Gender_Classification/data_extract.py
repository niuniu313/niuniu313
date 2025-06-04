import os
import pandas as pd

# 生成 dataset.csv 文件
image_dir = 'data\\part2'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

data = []
for file in image_files:
    try:
        parts = file.split('_')
        gender = int(parts[1])  # 文件名中第2部分是性别
        data.append({'file': file, 'gender': gender})
    except:
        continue

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
print(f"Saved dataset.csv with {len(df)} entries.")

