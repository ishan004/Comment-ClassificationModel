
import numpy as np

class MyBinaryRelevance:
    def __init__(self, classifier, max_iterations=10):
        self.classifier = classifier
        self.models = []
        self.max_iterations = max_iterations
        self.history = {"loss": []} # Initialize history dictionary

    def fit(self, X_train, Y_train):
        num_labels = Y_train.shape[1]

        for i in range(num_labels):
            model = self.classifier(kernel='linear')
            model.fit(X_train, Y_train[:, i])
            self.models.append(model)

            if i + 1 == self.max_iterations:
                break

    def predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_test)

        return predictions