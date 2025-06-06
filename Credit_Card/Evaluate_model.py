import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, X, y):
    """
    Evaluates a machine learning model using various metrics for both classification and regression.
    
    Parameters:
    -----------
    model_name : str
        Name of the model for display purposes
    model : estimator object
        The machine learning model to evaluate
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test target values
    X, y : array-like
        Complete dataset for cross-validation
        
    Returns:
    --------
    pipeline : Pipeline
        Fitted scikit-learn pipeline with the model
    """
    # Create a pipeline with just this model
    pipeline = Pipeline(steps=[('model', model)])
    
    print(f"Training {model_name}...")
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Predict
    pred = pipeline.predict(X_test)
    
    # Classification metrics
    acc = accuracy_score(y_test, pred)
    print(f"{model_name}:")
    print(f"  Accuracy: {acc:.4f}")
    
    # Cross-validation score for classification
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    cv_acc = cv_scores.mean()
    print(f"  CV Accuracy: {cv_acc:.4f}")
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    # Cross-validation score for regression
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV RMSE: ${cv_rmse:.2f}")
    
    print("-----------------------------")
    
    return pipeline

# Example usage:
model = RandomForestRegressor()
results = evaluate_model("Random Forest", model, X_train, X_test, y_train, y_test, X, y)

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
pipeline = evaluate_all_models(X_train, X_test, y_train, y_test, X, y)