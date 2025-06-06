import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

# Define the steps (list of model tuples)
steps = [
    ('logistic_regression', LogisticRegression()),
    ('ridge_classifier', RidgeClassifier(alpha=1.0)),
    ('lasso_regression', Lasso(alpha=0.01, max_iter=10000)),
    ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(kernel='rbf', C=1.0)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('xgboost', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('lightgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, force_col_wise=True))
]

# Function to evaluate models
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, X, y):
    # Create a pipeline with just this model
    pipeline = Pipeline(steps=[('model', model)])
    
    print(f"Training {model_name}...")
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Predict
    pred = pipeline.predict(X_test)
    
    # Evaluate
    # For classification models
    if hasattr(pipeline, "predict_proba") or model_name in ['logistic_regression', 'ridge_classifier', 'random_forest', 
                                            'gradient_boosting', 'svc', 'knn', 'xgboost', 'lightgbm']:
        acc = accuracy_score(y_test, pred)
        print(f"{model_name}:")
        print(f"  Accuracy: {acc:.4f}")
        
        # Cross-validation score for classification
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        cv_acc = cv_scores.mean()
        print(f"  CV Accuracy: {cv_acc:.4f}")
    
    # For regression models
    else:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        # Cross-validation score for regression
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        print(f"{model_name}:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  CV RMSE: ${cv_rmse:.2f}")
    
    print("-----------------------------")
    
    return pipeline

# Example usage (assuming you have X and y):
# X, y = load_your_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate all models
def evaluate_all_models(X_train, X_test, y_train, y_test, X, y):
    results = {}
    
    for name, model in steps:
        try:
            print(f"\nEvaluating {name}...")
            pipeline = evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y)
            results[name] = pipeline
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
    
    return results

# Call this function with your data:
# results = evaluate_all_models(X_train, X_test, y_train, y_test, X, y)