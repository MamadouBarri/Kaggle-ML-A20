import numpy as np
from sklearn.neural_network import MLPClassifier

class neural_network:

    def __init__(self, learning_rate, hidden_layers_sizes, data, labels):
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layers_sizes
        self.nn = MLPClassifier(alpha=self.learning_rate, solver='sgd', shuffle=True, hidden_layer_sizes=self.hidden_layer_sizes)
        self.nn.fit(data, labels)

    def train(self, data, labels):
        self.nn.fit(data, labels)

    def predict(self, data):
        return self.nn.predict_proba(data)

    def measure_classification(self, test_data, test_labels):

        if len(test_data) != len(test_labels):
            raise Exception('Data and labels must be the same size')
        
        return self.nn.score(test_data, test_labels)