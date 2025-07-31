
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.metrics import record_metrics

class Client:
    def __init__(self, client_id, dataset, model_fn, device='cpu'):
        self.id = client_id
        self.device = device
        self.model = model_fn().to(device)
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs=1, batch_size=32):
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        total_loss, correct = 0, 0
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(y).sum().item()
        acc = correct / len(loader.dataset)
        return self.model.state_dict(), total_loss, acc
