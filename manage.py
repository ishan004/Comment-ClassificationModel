#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import numpy as np

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SVM.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
    
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
    
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  
        self.class_probs = None
        self.feature_probs = None

    # def fit(self, X, y):
    #     num_classes = len(np.unique(y))
    #     num_features = X.shape[1]

    #     self.class_probs = np.zeros(num_classes)
    #     self.feature_probs = np.zeros((num_classes, num_features))

    #     for c in range(num_classes):
    #         class_mask = (y == c)
    #         class_count = np.sum(class_mask)
            
           
    #         self.class_probs[c] = (class_count + self.alpha) / (len(y) + num_classes * self.alpha)

            
    #         feature_counts = np.sum(X[class_mask], axis=0)

    #         # prob of feature given class
    #         self.feature_probs[c] = (feature_counts + self.alpha) / (class_count + num_features * self.alpha)

    def predict_proba(self, X):
        num_classes = self.feature_probs.shape[0]
        log_probs = np.zeros((X.shape[0], num_classes))

        for c in range(num_classes):
            #log  prob
            log_probs[:, c] = np.log(self.class_probs[c]) + np.sum(X * np.log(self.feature_probs[c] + 1e-10), axis=1)

        # normal prob
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == '__main__':
    main()
