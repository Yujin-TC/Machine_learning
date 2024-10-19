from decision_tree import ID3, predict_all
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# build a bagging tree using ID3 simple decision tree
def adaboost_tree_model(df, label, n_trees, n_samples, method='entropy'):
    # n_samples=1000
    trees = []
    n_samples, n_features = df.shape
    # df has feature + response, so substract one
    n_features = n_features-1
    # Initialize sample weights to 1/N
    sample_weights = np.ones(n_samples) / n_samples
    for i in range(n_trees):
        decision_tree = ID3(df, label, method, max_depth=1)
        trees.append(decision_tree)
    return trees


class adaboost_tree:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, df, label):
        y = df[[label]]
        X = df.drop(columns=[label])
        n_samples, n_features = X.shape

        # Initialize sample weights to 1/N
        # Initialize sample weights to 1/N
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train a weak learner (decision stump) with the current sample weights
            stump = self.base_estimator
            stump.fit(X, y, sample_weight=sample_weights)
            predictions = stump.predict(X)

            # Compute the weighted error
            error = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights)

            # Avoid division by zero or log of zero
            if error == 0:
                error = 1e-10

            # Compute the model weight (alpha)
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights: increase for misclassified samples
            sample_weights *= np.exp(-alpha * y * predictions)

            # Normalize the weights to sum to 1
            sample_weights /= np.sum(sample_weights)

            # Store the stump and its corresponding alpha
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        # Initialize the final prediction as zero
        final_prediction = np.zeros(X.shape[0])

        # Aggregate predictions from each weak learner
        for model, alpha in zip(self.models, self.alphas):
            predictions = model.predict(X)
            final_prediction += alpha * predictions

        # Return the sign of the final aggregated prediction
        return np.sign(final_prediction)

