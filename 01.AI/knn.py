import numpy as np
import pandas as pd
import joblib
import os

# Tạo thư mục models nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

df = pd.read_csv("data.csv")
X = df.drop(columns='Outcome').values
y = df['Outcome'].values

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Lưu X, y để sau này dự đoán
joblib.dump((X, y), "models/knn_data.pkl")
joblib.dump(mean, "models/mean.npy")
joblib.dump(std, "models/std.npy")
print("Train & save KNN hoàn tất.")
