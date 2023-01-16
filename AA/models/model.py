import numpy as np

def compute_loss(outputs, labels):
    return -np.mean(labels * np.log(outputs) + (1 - labels) * np.log(1 - outputs))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, in_features, lr=0.001):
        self.weights = np.random.randn(in_features)
        self.lr = lr
        self.grads = None
        
    def forward(self,
                input_features,
                labels=None,
        ):
        outputs = np.matmul(input_features, self.weights)
        if labels is not None:
            loss = compute_loss(outputs, labels)
            self.grads = (sigmoid(outputs) - labels) * input_features
            return loss, outputs
        else:
            return outputs
        
    def optimize(self):
        self.weights -= self.lr * self.grads
        
    def zero_grad(self):
        self.grads = None