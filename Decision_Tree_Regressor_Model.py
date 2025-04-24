import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('processed_california_housing.csv')
X = df.drop('Target', axis=1)
Y = df['Target']

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)

tree_parameters = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
decisiontree_regressor = DecisionTreeRegressor(random_state=42)
decisiontree_regressor_grid = GridSearchCV(decisiontree_regressor, tree_parameters, cv=5, scoring='neg_mean_squared_error')
decisiontree_regressor_grid.fit(X_train, Y_train)

print("Best parameters for Decision Tree Regressor:", decisiontree_regressor_grid.best_params_)
print("Decision tree evaluation on validation set: ")
dtr_best = decisiontree_regressor_grid.best_estimator_

Y_pred = dtr_best.predict(X_val)
mae = mean_absolute_error(Y_val, Y_pred)
mse = mean_squared_error(Y_val, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_val, Y_pred)

print(f"Decision Tree Regressor Mean Absolute Error: {mae:.4f}")
print(f"Decision Tree Regressor Mean Squared Error: {mse:.4f}")
print(f"Decision Tree Regressor Root Mean Absolute Error: {rmse:.4f}")
print(f"Decision Tree Regressor R2 Score: {r2:.4f}")
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(dtr_best, X_train, Y_train, scoring='r2', cv=kfolds)
print(f"Decision Tree Regressor Cross-Validation R2 scores: {score}")
print(f"Decision Tree Regressor Mean CV R2: {np.mean(score):.4f}, Std: {np.std(score):.4f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(Y_val, Y_pred, alpha=0.6)
plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], color='red')
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

importance = dtr_best.feature_importances_
f = X.columns
plt.figure(figsize=(6,6))
plt.barh(f, importance)
plt.xlabel('Feature Importance')
plt.title("Decision Tree Feature Importances")
plt.show()