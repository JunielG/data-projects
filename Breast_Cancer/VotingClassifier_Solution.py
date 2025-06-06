from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create individual models
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42)
svc = SVC(probability=True, random_state=42)

# Create VotingClassifier with proper format: [(name, estimator), ...]
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('svc', svc)
])

# Now fit the classifier
voting_classifier.fit(X_train, y_train)

Each estimator needs a name and an actual model object, not just a number

### VotingClassifier Overview
The VotingClassifier is an ensemble learning method in scikit-learn that combines multiple machine learning classifiers and uses a voting strategy to make the final prediction. It's based on the idea that combining different models often leads to better overall performance than any single model alone.

### How It Works
There are two main voting strategies:

Hard Voting (majority voting): Each classifier votes for a class, and the class that receives the most votes becomes the final prediction.
Soft Voting (weighted probability): Each classifier provides a probability for each class, these probabilities are averaged (or weighted), and the class with the highest average probability is selected.

### Key Parameters
estimators: List of tuples (name, estimator), where:
- name is a unique string identifier for the estimator
- estimator is a fitted classifier object that implements .predict() and .predict_proba()

voting: String, either 'hard' or 'soft' (default='hard')
For soft voting, all classifiers must support predict_proba()
weights: List of weights for each classifier (default=None, which means equal weights)

Useful when some classifiers perform better than others

### Example with Different Voting Methods

# Create a VotingClassifier with hard voting
hard_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
    voting='hard'
)

# Create a VotingClassifier with soft voting
soft_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
    voting='soft'
)

# With weighted voting (giving more importance to RandomForest)
weighted_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
    voting='soft',
    weights=[2, 1, 1]  # RF has double the weight
)