# Check the data types and unique values
print(f"y_test_rf type: {type(y_test_rf)}")
print(f"rf_pred type: {type(rf_pred)}")

# Look at some samples
print("First few values of y_test_rf:", y_test_rf[:5])
print("First few values of rf_pred:", rf_pred[:5])

# Check for any continuous values
import numpy as np
print("Are there any float values in y_test_rf?", any(isinstance(x, float) and not x.is_integer() for x in y_test_rf))
print("Are there any float values in rf_pred?", any(isinstance(x, float) and not x.is_integer() for x in rf_pred))

# If needed, convert to integer class labels
y_test_rf_fixed = np.round(y_test_rf).astype(int)
rf_pred_fixed = np.round(rf_pred).astype(int)

# Try the classification report again
from sklearn.metrics import classification_report
print(classification_report(y_test_rf_fixed, rf_pred_fixed))