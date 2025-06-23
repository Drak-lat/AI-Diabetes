import numpy as np
import pandas as pd
import joblib
import os
from decision_stump import DecisionStump    # Import đúng class dùng chung!

# Tạo thư mục models nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

# Đọc dữ liệu
df = pd.read_csv("data.csv")
X = df.drop(columns='Outcome').values
y = df['Outcome'].values

# Chuẩn hóa
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Huấn luyện Random Forest 10 cây
trees = []
for _ in range(10):
    idxs = np.random.choice(len(X), len(X), replace=True)
    stump = DecisionStump()
    stump.train(X[idxs], y[idxs])
    trees.append(stump)

joblib.dump(trees, "models/rf_trees.pkl")
joblib.dump(mean, "models/mean.npy")
joblib.dump(std, "models/std.npy")
print("Train & save Random Forest giản lược hoàn tất.")
