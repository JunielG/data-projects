from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'n_estimators': [1, 10, 20, 30, 40, 50, 100]}

# Create and fit model
adaboost = AdaBoostClassifier()
grid_search = GridSearchCV(adaboost, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")