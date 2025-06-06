from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

estimators = [1, 10, 20, 30, 40, 50, 100]
scores = []

for n_est in estimators:
    adaboost = AdaBoostClassifier(n_estimators=n_est)
    adaboost.fit(X_train, y_train)
    score = adaboost.score(X_test, y_test)
    scores.append(score)
    print(f"n_estimators={n_est}, accuracy={score:.4f}")