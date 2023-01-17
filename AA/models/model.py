import numpy as np

def compute_loss(outputs, labels, eps=1e-5):
    if outputs == labels:
        return 0
    return -(labels * np.log(outputs + eps) + (1 - labels) * np.log(1 - outputs + eps))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, in_features, lr=0.001):
        self.weights = np.random.rand(in_features)
        self.lr = lr
        self.grads = None
        self.training = True
        
    def forward(self,
                input_features,
                labels=None,
        ):
        outputs = np.matmul(input_features, self.weights)
        outputs = sigmoid(outputs)
        if labels is not None:
            loss = compute_loss(outputs, labels)
            if self.training:
                self.grads = (outputs - labels) * input_features
            return loss, outputs
        else:
            return outputs
        
    def optimize(self):
        self.weights -= self.lr * self.grads
        
    def zero_grad(self):
        self.grads = None
        
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True