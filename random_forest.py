import scipy.io as sp
from sklearn import ensemble
from sklearn import tree
import numpy as np


def class_probs(Y):
    """
    Calculate the class probabilities by counting the unique classes.

    Args:
        Y: Class labels of the dataset.

    Returns:
        classes: unique class labels
        probs: proportion of each class in Y
    """
    classes, counts = np.unique(Y, return_counts=True)
    probs = counts / len(Y)
    return classes, probs


class Criterion:
    """Holds functions for calculating information of a decision tree node"""

    def get(self, name):
        return {'gini': self.gini_impurity, 'entropy': self.entropy}[name]

    @staticmethod
    def gini_impurity(Y):
        """The gini impurity approximates the probability of misclassifying a
        randomly sampled example in a decision node."""
        _, probs = class_probs(Y)
        return 1 - np.sum(np.square(probs))

    @staticmethod
    def entropy(Y):
        _, probs = class_probs(Y)
        return -np.sum(probs * np.log2(probs))


class Decision_node:
    def __init__(self, false_child, true_child, question):
        """
        The function of a decision node is to ask a question that maximize the
        information gain of the decision tree.

        Args:
            false_child: an Decision_Node or Leaf object, split by the parent question.
            true_child: an Decision_Node or Leaf object, split by the parent question.
            question: a Question object.
        """
        self.false_child = false_child
        self.true_child = true_child
        self.question = question


class Leaf:
    def __init__(self, Y):
        """
        The deepest node of a decision tree.

        Args:
            Y: Class labels of the dataset.

        Attributes:
            prediction: the class label decided by majority voting.  if two
            classes has the same class count/probabilities, the first class
            encountered is returned
        """
        classes, probs = class_probs(Y)
        max_idx = np.argmax(probs)
        self.prediction = classes[max_idx]


class Question:
    def __init__(self, value, column):
        """
        A question is an object that contains a splitting condition of feature vector(s).

        Args:
            value: int, used as the decision boundary between two categorical values of a feature.
            column: int, for remembering which feature to ask the question
        """
        self.decision_boundary = value
        self.column = column

    def __repr__(self):
        return 'decision_boundary={}, col={}'.format(self.decision_boundary, self.column)

    def ask(self, X):
        """Since pacman features are categorical, the question will be equalities,
        rather than inequalities"""
        is_true = X[:, self.column] == self.decision_boundary
        return is_true

    def split(self, X, Y):
        """Split the data by asking the question, so the data could be processed
        further down the tree branch."""
        is_true = self.ask(X)
        is_false = is_true == 0
        return X[is_false], Y[is_false], X[is_true], Y[is_true]


class DecisionTree:
    def __init__(self, criterion='gini'):
        """
        An implementation of CART adapted from:
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        https://github.com/random-forests/tutorials/blob/master/decision_tree.py

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        I have modified it into into the paradigm of 'declare, fit, predict',
        similar to sklearn. Furthermore, it was refactered to utilise numpy for
        speed, rather than using python lists. It is served as the building
        block for my random forest classifier.

        The criterion is a metric for measuring information gain:
        Possible values: 'gini' or 'entropy'"""
        self.criterion = Criterion().get(criterion)
        self.root = None

    def information_gain(self, false_Y, true_Y, parent_information):
        """Information gain is the difference between parent node information and child nodes information"""
        p_false = len(false_Y) / (len(false_Y) + len(true_Y))
        p_true = 1 - p_false
        # The new information is a weighted sum of child node's information.
        new_information = p_false * self.criterion(false_Y) + p_true * self.criterion(true_Y)
        gain = parent_information - new_information
        return gain

    def find_best_split(self, X, Y):
        """
        Finds the best gain and question among all possible features and their
        values at a decision node.

        Returns:
            best_gain: float
            best_question: Question object
        """
        best_gain = 0
        best_question = None
        n_rows, n_features = X.shape
        current_information = self.criterion(Y)

        for j in range(n_features):
            for i in range(n_rows):
                # ask the question for feature j and sample i
                question = Question(X[i, j], j)
                _, false_Y, _, true_Y = question.split(X, Y)

                # if a feature only has one value, then the split is invalid,
                # therefore we skip.
                if len(true_Y) == 0 or len(false_Y) == 0:
                    continue

                # keep track of the best gain and question
                gain = self.information_gain(false_Y, true_Y, current_information)
                if gain > best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def build_decision_tree(self, X, Y):
        """Recursively build the decision tree."""
        # Iterate through all possible features and their values, and find the
        # question with best information gain
        gain, question = self.find_best_split(X, Y)

        # No information gain, meaning no better question/prediction can be made.
        if gain <= 0:
            return Leaf(Y)

        # if a positive gain is found, then go further down the branches, until
        # no further info gain.
        false_X, false_Y, true_X, true_Y = question.split(X, Y)
        false_child = self.build_decision_tree(false_X, false_Y)
        true_child = self.build_decision_tree(true_X, true_Y)

        return Decision_node(false_child, true_child, question)

    def fit(self, X, Y):
        """
        Args:
            X: numpy array of shape [n_samples, n_features]
            Y: numpy array of shape [n_samples]
        """
        self.root = self.build_decision_tree(X, Y)

    def predict_sample(self, x):
        """
        Args:
            x: a single sample vector of shape [1, n_features] or [n_features]
        """

        # Convert x to at least [1, n_features], because Question objects at
        # each decision node needs to access its column dimension.
        x = np.atleast_2d(x)

        node = self.root

        # Traverse down the tree until reaching a leaf, then make a prediction
        # based on majority voting.
        while True:
            if isinstance(node, Leaf):
                return node.prediction

            is_true = node.question.ask(x)

            if is_true:
                node = node.true_child
            else:
                node = node.false_child

    def predict(self, X):
        """
        Args:
            X: feature vectors of shape [n_samples, n_features]
        """
        return np.apply_along_axis(self.predict_sample, axis=1, arr=X)

    def score(self, X, Y):
        """Accuracy score"""
        return (self.predict(X) == Y).mean()


class RandomForest:
    def __init__(self, n_estimators=100, criterion='gini'):
        """
        A basic implementation of random forest
        It bootstraps all the samples as well as sqrt(n_feature) features with replacement.

        Args:
            n_estimators: number of trees in the forest
            criterion: 'gini' or 'entropy', metric for measuring information gain

        Attributes:
            forest: a list of decision trees
            forest_features: a list of feature indices for remembering what
                    feature subspace are bootstrapped for each tree.
        """

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.forest = []
        # remembers what features subspace are selected for each tree
        self.forest_features = []

        # shape of data X
        self.n_samples = None
        self.n_features = None

    def bootstrap(self, X, Y):
        """Bootstrapping all the data points with replacement."""
        sample_indices = np.arange(self.n_samples)
        bootstrap_sample_indices = np.random.choice(sample_indices, size=self.n_samples, replace=True)
        # print(np.unique(bootstrap_sample_indices).size/len(Y)) # (diagnostic) should be about 0.63
        return X[bootstrap_sample_indices], Y[bootstrap_sample_indices]

    def random_subspace(self, X):
        """
        Sample a random subset of the features

        Returns:
            X_subspace: data with bootstrapped features
            selected_features: array of selected feature indices
        """
        max_features = int(np.floor(np.sqrt(self.n_features)))
        feature_indices = np.arange(self.n_features)
        selected_features = np.random.choice(feature_indices, size=max_features, replace=True)
        X_subspace = X[:, selected_features]
        return X_subspace, selected_features

    def fit(self, X, Y):
        """
        Fit self.n_estimators decision trees, each trained with a bootstrapped samples and features.
        Populates self.forest and self.forest_features in the process.

        Args:
            X: numpy array of shape [n_samples, n_features]
            Y: numpy array of shape [n_samples]
        """
        self.n_samples, self.n_features = X.shape

        for _ in range(self.n_estimators):
            # Impose a minimum tree depth > 1
            while True:
                try:
                    x, selected_features = self.random_subspace(X)
                    x, y = self.bootstrap(x, Y)
                    tree = DecisionTree(self.criterion)
                    tree.fit(x, y)

                    # if the root is not a decision node, then it means the
                    # depth = 1
                    assert isinstance(tree.root, Decision_node)
                    break
                except AssertionError:
                    print('Retrying a different feature subset until tree depth > 1')

            self.forest.append(tree)
            self.forest_features.append(selected_features)

    def predict_sample(self, x):
        """Predict by majority voting by the ensemble of trees."""
        # aggregating predictions of all the trees
        forest_preds = [
            tree.predict_sample(x[selected_features])
            for tree, selected_features in zip(self.forest, self.forest_features)
        ]
        # majority voting
        return np.bincount(forest_preds).argmax()

    def predict(self, X):
        """Takes feature vectors of shape [n_samples, n_features]"""
        return np.apply_along_axis(self.predict_sample, axis=1, arr=X)

    def score(self, X, Y):
        """Accuracy score"""
        return (self.predict(X) == Y).mean()


if __name__ == "__main__":
    data = sp.loadmat('iris_class1_2_3_4D.mat')

    Y = np.ravel(data['t'])
    X = data['X'].T
    print(X.shape)

    clf = RandomForest(n_estimators=20, criterion='entropy')
    clf2 = ensemble.RandomForestClassifier(n_estimators=20)
    # clf = DecisionTree('entropy')
    # clf2 = tree.DecisionTreeClassifier()

    clf.fit(X, Y)
    # print(clf.predict(X))
    # print(Y)
    print(clf.score(X, Y))

    clf2.fit(X, Y)
    # print(clf.predict(X))
    # print(Y)
    print(clf2.score(X, Y))

    print((Y == np.bincount(Y).argmax()).mean())
