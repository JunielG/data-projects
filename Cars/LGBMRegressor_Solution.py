# Assuming you have these variables:
# - model: your trained LGBMRegressor model
# - X_train: the DataFrame you used for training
# - X_pred: the data you're using for prediction (causing the warning)

# Solution 1: If X_pred is a numpy array or doesn't have column names
# Convert X_pred to a DataFrame with the same column names as X_train
import pandas as pd

# Get feature names from the training data
feature_names = X_train.columns.tolist()

# Convert prediction data to DataFrame with matching feature names
X_pred_df = pd.DataFrame(X_pred, columns=feature_names)

# Now use this for prediction
predictions = model.predict(X_pred_df)

# Solution 2: If X_pred is already a DataFrame but with different/missing column names
# Make sure X_pred has the same columns in the same order
X_pred_fixed = X_pred.reindex(columns=X_train.columns)

# Now predict
predictions = model.predict(X_pred_fixed)


### Code 2 
Use a pandas DataFrame for prediction:
# Instead of:
y_pred = model.predict(X_test)  # If X_test is a numpy array

# Do:
import pandas as pd
X_test_df = pd.DataFrame(X_test, columns=feature_names)
y_pred = model.predict(X_test_df)

Or, if you want to stick with numpy arrays, disable the warning:
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
y_pred = model.predict(X_test)

