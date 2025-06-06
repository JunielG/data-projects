from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
import numpy as np

# Assuming you already have your model trained
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Calculate permutation importance
result = permutation_importance(
    svr_model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

# Get the importance values
importance = result.importances_mean

# Create a dictionary mapping feature names to importance values
feature_importance = {feature_name: imp for feature_name, imp in zip(X.columns, importance)}

# Sort features by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Display feature importance
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")