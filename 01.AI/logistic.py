import numpy as np
import pandas as pd
import joblib
import os

# Tạo thư mục models nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

# Load data
df = pd.read_csv("data.csv")
X = df.drop(columns='Outcome').values
y = df['Outcome'].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Thêm cột bias
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Sigmoid + Loss
def sigmoid(z): return 1 / (1 + np.exp(-z))
def loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1 - y_pred))

# Huấn luyện
def train(X, y, lr=0.1, epochs=1000):
    w = np.zeros((X.shape[1], 1))
    for epoch in range(epochs):
        z = X @ w
        y_pred = sigmoid(z)
        grad = X.T @ (y_pred - y) / len(y)
        w -= lr * grad
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss(y, y_pred):.4f}")
    return w

w = train(X, y)

# Lưu mô hình và thông tin chuẩn hóa
joblib.dump(w, "models/logistic_manual.pkl")
joblib.dump(mean, "models/mean.npy")
joblib.dump(std, "models/std.npy")
print("Train & save logistic regression hoàn tất.")
