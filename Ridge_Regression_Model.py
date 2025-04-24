import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('processed_california_housing.csv')
X = df.drop('Target', axis=1)
Y = df['Target']


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)

ridge_parameters = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_regressor = Ridge(random_state=42)
ridge_regressor_grid = GridSearchCV(ridge_regressor, ridge_parameters, cv=5, scoring='neg_mean_squared_error')
ridge_regressor_grid.fit(X_train, Y_train)
rr_best = ridge_regressor_grid.best_estimator_

print("Best parameters for Ridge Regression:", ridge_regressor_grid.best_params_)
print("Ridge Regression Evaluation on Validation Set: ")

Y_pred = rr_best.predict(X_val)
mae = mean_absolute_error(Y_val, Y_pred)
mse = mean_squared_error(Y_val, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_val, Y_pred)

print(f"Ridge Regression Mean Absolute Error: {mae:.4f}")
print(f"Ridge Regression Mean Squared Error: {mse:.4f}")
print(f"Ridge Regression Root Mean Absolute Error: {rmse:.4f}")
print(f"Ridge Regression R2 Score: {r2:.4f}")

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(rr_best, X_train, Y_train, scoring='r2', cv=kfolds)
print(f"Ridge Regression Cross-Validation R2 scores: {score}")
print(f"Ridge Regression Mean CV R2: {np.mean(score):.4f}, Std: {np.std(score):.4f}")


plt.figure(figsize=(8, 5))
plt.scatter(Y_val, Y_pred, alpha=0.6)
plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], color='red')
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Ridge Regression: Actual vs Predicted")
plt.grid(True)
plt.show()