# Define features - use column names instead of DataFrame objects
numeric_features = ['Year', 'Engine_Size_Liter']  # Removed Price_USD as it's the target
categorical_features = ['Make', 'Model', 'Fuel_Type']
X = df[numeric_features + categorical_features]
y = df['Price_USD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Function to evaluate models
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Predict
    pred = pipeline.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"{model_name}:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV RMSE: ${cv_rmse:.2f}")
    print("-----------------------------")
    
    return pipeline, rmse, r2, cv_rmse

# Define models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Evaluate all models
results = {}
best_rmse = float('inf')
best_model_name = None
best_pipeline = None
for name, model in models.items():
    try:
        print(f"Training {name}...")
        pipeline, rmse, r2, cv_rmse = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = {'rmse': rmse, 'r2': r2, 'cv_rmse': cv_rmse, 'pipeline': pipeline}
        
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model_name = name
            best_pipeline = pipeline
            
    except Exception as e:
        print(f"Error with {name}: {e}")
print(f"\nBest model: {best_model_name} with CV RMSE: ${best_rmse:.2f}")