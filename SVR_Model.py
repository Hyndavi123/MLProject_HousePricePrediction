import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed dataset
df = pd.read_csv("processed_california_housing.csv")

# Split features and target
X = df.drop(columns="Target")
y = df["Target"]

# Split into training (60%), validation (20%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SVR parameter tuning
param_grid = {
    'C': [1, 10],
    'epsilon': [0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
grid_svr = GridSearchCV(SVR(), param_grid, cv=5)
grid_svr.fit(X_train, y_train)

# Use the best estimator
svr_best = grid_svr.best_estimator_
y_pred = svr_best.predict(X_val)

# Evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
cross_val = np.mean(cross_val_score(svr_best, X_train, y_train, cv=5))

# Print metrics
print("Support Vector Regression Metrics:")
print(f"Best Params: {grid_svr.best_params_}")
print(f"Mean Absolute Error  : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.4f}")
print(f"Root Mean Squared Error : {rmse:.4f}")
print(f"R2 score   : {r2:.4f}")
print(f"Cross-Validation Score: {cross_val:.4f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("SVR: Actual vs Predicted")
plt.grid(True)
plt.show()