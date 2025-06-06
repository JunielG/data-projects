from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Define your models
steps = [
    ('logistic_regression', LogisticRegression()),
    ('ridge_classifier', RidgeClassifier(alpha=1.0)),
    # Note: Lasso and ElasticNet are regression models, not classifiers
    # They won't work with accuracy_score which is for classification
    # ('lasso_regression', Lasso(alpha=0.01, max_iter=10000)),
    # ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(kernel='rbf', C=1.0)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('xgboost', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('lightgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, force_col_wise=True))
]

# Sample data for demonstration (replace with your actual data)
# X = your_features_data
# y = your_target_data

# For demonstration purposes:
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in steps:
    try:
        print(f"Training {name}...")
        # Create a pipeline with just this model
        pipeline = Pipeline(steps=[(name, model)])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"  Accuracy Score for {name}: {acc:.4f}")
    except Exception as e:
        print(f"Error with {name}: {e}")

# Sort and display results
print("\nModel Performance Summary (sorted by accuracy):")
results_df = pd.DataFrame({'Model': list(results.keys()), 'Accuracy': list(results.values())})
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print(results_df)

# Optionally, create a simple bar chart
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    
    # Add the accuracy values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.show()
except ImportError:
    print("Matplotlib not available for visualization")