from decision_stump import DecisionStump

import numpy as np
import joblib

mean = joblib.load("models/mean.npy")
std = joblib.load("models/std.npy")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic(X_input):
    w = joblib.load("models/logistic_manual.pkl")
    X_bias = np.insert(X_input, 0, 1)
    proba = sigmoid(np.dot(X_bias, w))[0]
    return proba

def predict_knn(X_input, k=5):
    X_train, y_train = joblib.load("models/knn_data.pkl")
    dists = np.linalg.norm(X_train - X_input, axis=1)
    idx = np.argsort(dists)[:k]
    proba = np.mean(y_train[idx])
    return proba

def predict_rf(X_input):
    trees = joblib.load("models/rf_trees.pkl")
    votes = [tree.predict(X_input[0]) for tree in trees]
    proba = np.mean(votes)
    return proba

def predict_soft(input_data):
    X_norm = (input_data - mean) / std
    p1 = predict_logistic(X_norm[0])
    p2 = predict_knn(X_norm[0])
    p3 = predict_rf(X_norm)
    final_prob = (p1 + p2 + p3) / 3
    return {
        'logistic': p1,
        'knn': p2,
        'rf': p3,
        'soft_voting': final_prob
    }

if __name__ == "__main__":
    sample = np.array([[1, 89, 70, 23, 94, 28.1, 0.167, 21]])
    result = predict_soft(sample)
    print("Kết quả dự đoán:")
    print(f"Logistic:     {result['logistic']*100:.2f}%")
    print(f"KNN:          {result['knn']*100:.2f}%")
    print(f"RandomForest: {result['rf']*100:.2f}%")
    print(f"Soft voting:  {result['soft_voting']*100:.2f}%")
