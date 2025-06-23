import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.pred_left = None
        self.pred_right = None

    def train(self, X, y):
        n_features = X.shape[1]
        best_gini = float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                left_gini = 1.0 - sum((np.mean(y[left_mask] == c))**2 for c in [0,1])
                right_gini = 1.0 - sum((np.mean(y[right_mask] == c))**2 for c in [0,1])
                gini = (sum(left_mask) * left_gini + sum(right_mask) * right_gini) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    self.feature = feature
                    self.threshold = threshold
                    self.pred_left = round(np.mean(y[left_mask]))
                    self.pred_right = round(np.mean(y[right_mask]))
    def predict(self, x):
        return self.pred_left if x[self.feature] <= self.threshold else self.pred_right
