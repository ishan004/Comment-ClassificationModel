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


if __name__ == '__main__':
    main()
