### Increase max_iter significantly (try 10000 or more):
Lasso(alpha=0.01, max_iter=10000)

### Try standardizing your features first (if you haven't already):
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Adjust the alpha parameter (try different values):
Lasso(alpha=0.1, max_iter=1000)  # Larger alpha

### Set a higher tolerance:
Lasso(alpha=0.01, max_iter=1000, tol=1e-2)