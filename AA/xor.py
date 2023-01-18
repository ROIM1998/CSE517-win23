import torch
import torch.nn as nn
from torch.optim import AdamW

class XORModel(nn.Module):
    def __init__(self, hidden_dim=4) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, label):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.classifier(x))
        loss = nn.BCELoss()(x, label.unsqueeze(dim=0))
        return loss, x
    
def test(model, dataset):
    acc = 0
    for inputs, labels in dataset:
        loss, x = model(inputs, labels)
        prediction = torch.round(x)
        prediction = -1 if prediction == 0 else 1
        if prediction == labels:
            acc += 1
    return acc / len(dataset)

if __name__ == '__main__':
    model = XORModel(3)
    dataset = [
        ([0, 0], -1),
        ([1, 1], -1),
        ([1, 0], 1),
        ([0, 1], 1),
    ]
    dataset = [(torch.tensor(x).float(), torch.tensor(y).float()) for x, y in dataset]
    optimizer = AdamW(model.parameters(), lr=0.1)
    for i in range(100):
        for inputs, labels in dataset:
            loss, x = model(inputs, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            
            
    print("Training finished")
    print("Accuracy: %f" % test(model, dataset))